import math
import torch

from torch import nn


def generate_mask(n_pix):
    mask = torch.tril(torch.ones(2 * n_pix, 2 * n_pix))
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


def generate_multi_mask(n_pix):
    mask = torch.tril(torch.ones(2 * n_pix + 2, 2 * n_pix + 2))
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


class DEformer(nn.Module):
    def __init__(
        self,
        img_size,
        pos_in_feats,
        pixel_n,
        mlp_layers,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
    ):
        super().__init__()
        initrange = 0.1

        pos_mlp = nn.Sequential()
        pix_mlp = nn.Sequential()
        pix_in_feats = pos_in_feats + 1
        for (layer_idx, out_feats) in enumerate(mlp_layers):
            pos_mlp.add_module(f"layer{layer_idx}", nn.Linear(pos_in_feats, out_feats))
            pix_mlp.add_module(f"layer{layer_idx}", nn.Linear(pix_in_feats, out_feats))
            if layer_idx < len(mlp_layers) - 1:
                pos_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                pix_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

            pos_in_feats = out_feats
            pix_in_feats = out_feats

        self.pos_mlp = pos_mlp
        self.pix_mlp = pix_mlp

        d_model = mlp_layers[-1]
        self.d_model = d_model
        chns = 1 if pixel_n == 1 else 3
        n_pix = img_size ** 2 * chns
        self.register_buffer("mask", generate_mask(n_pix))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.pixel_classifier = nn.Linear(d_model, pixel_n)
        self.pixel_classifier.weight.data.uniform_(-initrange, initrange)
        self.pixel_classifier.bias.data.zero_()

    def forward(self, tensors):
        device = list(self.pixel_classifier.parameters())[0].device

        positions = tensors["positions"].to(device)
        pos_feats = self.pos_mlp(positions) * math.sqrt(self.d_model)
        pixels = tensors["pixels"].unsqueeze(1).to(device)
        pos_pixels = torch.cat([positions, pixels], dim=1)
        pix_feats = self.pix_mlp(pos_pixels) * math.sqrt(self.d_model)

        combined = torch.zeros(2 * len(pos_feats), self.d_model).to(device)
        combined[::2] = pos_feats
        combined[1::2] = pix_feats

        outputs = self.transformer(combined.unsqueeze(1), self.mask)
        preds = self.pixel_classifier(outputs.squeeze(1)[::2])

        return preds
