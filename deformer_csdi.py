import math
import torch

from torch import nn


class DEformerCSDI(nn.Module):
    def __init__(
        self,
        n_feats,
        idx_embed_dim,
        mlp_layers,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
    ):
        super().__init__()
        initrange = 0.1

        self.idx_embedding = nn.Embedding(n_feats, idx_embed_dim)

        mlps = {}
        for which_mlp in ["obs", "unobs"]:
            mlp = nn.Sequential()
            in_feats = idx_embed_dim + 2 if which_mlp == "obs" else idx_embed_dim + 1
            for (layer_idx, out_feats) in enumerate(mlp_layers):
                mlp.add_module(f"layer{layer_idx}", nn.Linear(in_feats, out_feats))
                if layer_idx < len(mlp_layers) - 1:
                    mlp.add_module(f"relu{layer_idx}", nn.ReLU())

                in_feats = out_feats

            mlps[which_mlp] = mlp

        self.mlps = nn.ModuleDict(mlps)

        d_model = mlp_layers[-1]
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.final_layer = nn.Linear(d_model, 1)
        self.final_layer.weight.data.uniform_(-initrange, initrange)
        self.final_layer.bias.data.zero_()

    def forward(self, tensors):
        device = list(self.final_layer.parameters())[0].device

        obs_vals = tensors["obs_vals"].unsqueeze(1).to(device)
        obs_times = tensors["obs_times"].unsqueeze(1).to(device)
        obs_feat_idx_embeds = self.idx_embedding(tensors["obs_feat_idxs"].to(device))
        obs_inputs = torch.cat([obs_vals, obs_times, obs_feat_idx_embeds], dim=1)
        obs_inputs = self.mlps["obs"](obs_inputs) * math.sqrt(self.d_model)

        unobs_times = tensors["unobs_times"].unsqueeze(1).to(device)
        unobs_feat_idx_embeds = self.idx_embedding(
            tensors["unobs_feat_idxs"].to(device)
        )
        unobs_inputs = torch.cat([unobs_times, unobs_feat_idx_embeds], dim=1)
        unobs_inputs = self.mlps["unobs"](unobs_inputs) * math.sqrt(self.d_model)

        all_inputs = torch.cat([obs_inputs, unobs_inputs]).unsqueeze(0).permute(1, 0, 2)

        outputs = self.transformer(all_inputs).permute(1, 0, 2).squeeze(0)
        n_obs = len(obs_inputs)
        preds = self.final_layer(outputs[n_obs:])

        return preds
