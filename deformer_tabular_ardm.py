import math
import torch

from torch import nn


def generate_mask(n_feats):
    mask = torch.tril(torch.ones(n_feats, n_feats))
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


class DEformerTabularARDM(nn.Module):
    def __init__(
        self,
        n_feats,
        idx_embed_dim,
        mix_comps,
        mlp_layers,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
    ):
        super().__init__()
        initrange = 0.1

        self.dummy_embedding = nn.Parameter(torch.randn(1, 1, idx_embed_dim + 1))
        self.idx_embedding = nn.Embedding(n_feats, idx_embed_dim)

        mlp = nn.Sequential()
        in_feats = 2 * idx_embed_dim + 1
        for (layer_idx, out_feats) in enumerate(mlp_layers):
            mlp.add_module(f"layer{layer_idx}", nn.Linear(in_feats, out_feats))
            if layer_idx < len(mlp_layers) - 1:
                mlp.add_module(f"relu{layer_idx}", nn.ReLU())

            in_feats = out_feats

        self.mlp = mlp

        d_model = mlp_layers[-1]
        self.d_model = d_model
        self.register_buffer("mask", generate_mask(n_feats))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Mixture of Gaussians parameters.
        self.mix_comps = mix_comps
        self.mix_predictor = nn.Linear(d_model, 3 * mix_comps)
        self.mix_predictor.weight.data.uniform_(-initrange, initrange)
        self.mix_predictor.bias.data.zero_()

    def forward(self, tensors):
        device = list(self.mix_predictor.parameters())[0].device

        idx_embeds = self.idx_embedding(tensors["idxs"].to(device))
        vals = tensors["vals"].unsqueeze(2).to(device)
        prev_idx_vals = torch.cat([idx_embeds[:, :-1], vals[:, :-1]], dim=2)
        rep_dummy = self.dummy_embedding.repeat(len(idx_embeds), 1, 1)
        prev_idx_vals = torch.cat([rep_dummy, prev_idx_vals], dim=1)
        combined = torch.cat([prev_idx_vals, idx_embeds], dim=2)
        combined = self.mlp(combined) * math.sqrt(self.d_model)
        combined = combined.permute(1, 0, 2)

        outputs = self.transformer(combined, self.mask).permute(1, 0, 2)
        preds = self.mix_predictor(outputs)

        mix_comps = self.mix_comps
        pis = torch.softmax(preds[..., :mix_comps], dim=2)
        mus = preds[..., mix_comps : 2 * mix_comps]
        log_sds = preds[..., 2 * mix_comps :]
        sds = torch.exp(log_sds)

        return (pis, mus, sds)
