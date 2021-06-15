import numpy as np
import scipy.io
import urllib.request
import torch

from torch.utils.data import Dataset


def download_notmnist():
    url_prefix = "http://yaroslavvb.com/upload/notMNIST"
    filename = "notMNIST_small.mat"
    url = f"{url_prefix}/{filename}"
    print(f"Downloading notMNIST from: {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Saved to {filename}.")
    data = scipy.io.loadmat(filename)["images"].transpose(2, 0, 1)
    # To plot one use: Image.fromarray(data[idx]).show().
    np.save("notMNIST.npy", data)
    binarized = np.array(data > 127, dtype="int8")
    # To plot one use: Image.fromarray(255 * binarized[idx]).show().
    np.save("binarized_notMNIST.npy", binarized)


class NotMNISTDataset(Dataset):
    def __init__(self):
        self.img_size = 28
        try:
            self.img_data = np.load("binarized_notMNIST.npy")
        except FileNotFoundError:
            download_notmnist()
            self.img_data = np.load("binarized_notMNIST.npy")

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        pixels = self.img_data[idx].flatten()
        idxs = np.arange(len(pixels))
        pixels = pixels[idxs]
        (rows, cols) = np.unravel_index(idxs, (self.img_size, self.img_size))
        positions = np.vstack([rows, cols]).T

        return {"pixels": torch.Tensor(pixels), "positions": torch.Tensor(positions)}
