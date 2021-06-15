import numpy as np
import torch

from torch.utils.data import Dataset


class DEformerDataset(Dataset):
    def __init__(self, dataset, img_data, is_train):
        assert dataset in {"mnist", "cifar10"}
        self.dataset = dataset
        self.img_size = 28 if dataset == "mnist" else 32
        self.img_data = img_data
        self.is_train = is_train

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        pixels = self.img_data[idx].flatten()
        idxs = np.arange(len(pixels))
        if self.is_train:
            np.random.shuffle(idxs)

        pixels = pixels[idxs]
        if self.dataset == "mnist":
            (rows, cols) = np.unravel_index(idxs, (self.img_size, self.img_size))
            positions = np.vstack([rows, cols]).T
        else:
            (rows, cols, chns) = np.unravel_index(
                idxs, (self.img_size, self.img_size, 3)
            )
            positions = np.vstack([rows, cols, chns]).T

        return {"pixels": torch.Tensor(pixels), "positions": torch.Tensor(positions)}
