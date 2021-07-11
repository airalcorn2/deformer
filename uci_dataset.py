import numpy as np
import pandas as pd
import shutil
import torch
import urllib.request

from torch.utils.data import Dataset


def get_correlation_numbers(data):
    C = data.corr()
    A = C > 0.98
    B = A.values.sum(axis=1)
    return B


def process_gas_data():
    # Adapted from: https://github.com/gpapamak/maf/blob/master/datasets/gas.py.
    # Dataset description: http://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+under+dynamic+gas+mixtures.

    data = pd.read_pickle("data/gas/ethylene_CO.pickle")
    data.drop("Meth", axis=1, inplace=True)
    data.drop("Eth", axis=1, inplace=True)
    data.drop("Time", axis=1, inplace=True)

    B = get_correlation_numbers(data)

    while np.any(B > 1):
        col_to_remove = np.where(B > 1)[0][0]
        col_name = data.columns[col_to_remove]
        data.drop(col_name, axis=1, inplace=True)
        B = get_correlation_numbers(data)

    data = (data - data.mean()) / data.std()

    N_test = int(0.1 * data.values.shape[0])
    data_test = data[-N_test:]
    data_train = data[:-N_test]
    N_validate = int(0.1 * data_train.shape[0])
    data_validate = data_train[-N_validate:]
    data_train = data_train[:-N_validate]

    np.save("gas_train.npy", data_train)
    np.save("gas_val.npy", data_validate)
    np.save("gas_test.npy", data_test)


def process_power_data():
    # Adapated from: https://github.com/gpapamak/maf/blob/master/datasets/power.py.
    # Dataset description: http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption.

    rng = np.random.RandomState(42)

    data = np.load("data/power/data.npy")
    rng.shuffle(data)
    N = data.shape[0]

    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)

    # Add noise.
    voltage_noise = 0.01 * rng.rand(N, 1)
    gap_noise = 0.001 * rng.rand(N, 1)
    sm_noise = rng.rand(N, 3)
    time_noise = np.zeros((N, 1))
    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data = data + noise

    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[:-N_validate]

    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    np.save("power_train.npy", data_train)
    np.save("power_val.npy", data_validate)
    np.save("power_test.npy", data_test)


def download_datasets():
    filename = "data.tar.gz"
    url = f"https://zenodo.org/record/1161203/files/{filename}?download=1"
    print(f"Downloading UCI datasets from: {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Saved to {filename}.")
    shutil.unpack_archive(filename, ".")
    process_gas_data()
    process_power_data()


class UCIDataset(Dataset):
    def __init__(self, dataset, train_val_test):
        assert dataset in {"power", "gas"}
        try:
            self.data = np.load(f"{dataset}_{train_val_test}.npy")
        except FileNotFoundError:
            download_datasets()
            self.data = np.load(f"{dataset}_{train_val_test}.npy")

        self.n_feats = self.data.shape[1]
        self.is_train = train_val_test == "train"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vals = self.data[idx]
        idxs = np.arange(len(vals))
        if self.is_train:
            np.random.shuffle(idxs)

        vals = vals[idxs]
        return {"idxs": torch.LongTensor(idxs), "vals": torch.Tensor(vals)}
