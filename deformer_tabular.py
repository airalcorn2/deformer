import math
import torch

from torch import nn


def generate_mask(n_feats):
    mask = torch.tril(torch.ones(2 * n_feats, 2 * n_feats))
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


class DEformerTabular(nn.Module):
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

        self.idx_embedding = nn.Embedding(n_feats, idx_embed_dim)

        idx_mlp = nn.Sequential()
        val_mlp = nn.Sequential()
        idx_in_feats = idx_embed_dim
        val_in_feats = idx_embed_dim + 1
        for (layer_idx, out_feats) in enumerate(mlp_layers):
            idx_mlp.add_module(f"layer{layer_idx}", nn.Linear(idx_in_feats, out_feats))
            val_mlp.add_module(f"layer{layer_idx}", nn.Linear(val_in_feats, out_feats))
            if layer_idx < len(mlp_layers) - 1:
                idx_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                val_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

            idx_in_feats = out_feats
            val_in_feats = out_feats

        self.idx_mlp = idx_mlp
        self.val_mlp = val_mlp

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
        idx_feats = self.idx_mlp(idx_embeds) * math.sqrt(self.d_model)
        vals = tensors["vals"].unsqueeze(2).to(device)
        idx_vals = torch.cat([idx_embeds, vals], dim=2)
        val_feats = self.val_mlp(idx_vals) * math.sqrt(self.d_model)

        combined = torch.zeros(len(idx_feats), 2 * len(idx_feats[0]), self.d_model).to(
            device
        )
        combined[:, ::2] = idx_feats
        combined[:, 1::2] = val_feats
        combined = combined.permute(1, 0, 2)

        outputs = self.transformer(combined, self.mask).permute(1, 0, 2)
        preds = self.mix_predictor(outputs[:, ::2])

        mix_comps = self.mix_comps
        pis = torch.softmax(preds[..., :mix_comps], dim=2)
        mus = preds[..., mix_comps : 2 * mix_comps]
        log_sds = preds[..., 2 * mix_comps :]
        sds = torch.exp(log_sds)

        return (pis, mus, sds)
