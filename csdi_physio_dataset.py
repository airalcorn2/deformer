# Adapted from: https://github.com/ermongroup/CSDI.

import numpy as np
import os
import pandas as pd
import re
import tarfile
import torch
import wget

from torch.utils.data import Dataset

# 35 attributes which contains enough non-values
attributes = [
    "DiasABP",
    "HR",
    "Na",
    "Lactate",
    "NIDiasABP",
    "PaO2",
    "WBC",
    "pH",
    "Albumin",
    "ALT",
    "Glucose",
    "SaO2",
    "Temp",
    "AST",
    "Bilirubin",
    "HCO3",
    "BUN",
    "RespRate",
    "Mg",
    "HCT",
    "SysABP",
    "FiO2",
    "K",
    "GCS",
    "Cholesterol",
    "NISysABP",
    "TroponinT",
    "MAP",
    "TroponinI",
    "PaCO2",
    "Platelets",
    "Urine",
    "NIMAP",
    "Creatinine",
    "ALP",
]


def download_data():
    os.makedirs("data/", exist_ok=True)
    url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
    wget.download(url, out="data")
    with tarfile.open("data/set-a.tar.gz", "r:gz") as t:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(t, path="data/physio")


def get_idlist():
    patient_id = []
    try:
        filenames = os.listdir("data/physio/set-a")
    except FileNotFoundError:
        download_data()
        filenames = os.listdir("data/physio/set-a")

    for filename in filenames:
        match = re.search("\d{6}", filename)
        if match:
            patient_id.append(match.group())

    patient_id = np.sort(patient_id)
    return patient_id


def extract_hour(x):
    (h, _) = map(int, x.split(":"))
    return h


def parse_data(x):
    # extract the last value for each attribute
    x = x.set_index("Parameter").to_dict()["Value"]

    values = []

    for attr in attributes:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)

    return values


def parse_id(id_, missing_ratio=0.1):
    data = pd.read_csv(f"data/physio/set-a/{id_}.txt")

    # set hour
    data["Time"] = data["Time"].apply(lambda x: extract_hour(x))

    # create data for 48 hours x 35 attributes
    observed_values = []
    for h in range(48):
        observed_values.append(parse_data(data[data["Time"] == h]))

    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)

    # randomly set some percentage as ground-truth
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(
        obs_indices, int(len(obs_indices) * missing_ratio), replace=False
    )
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return (observed_values, observed_masks, gt_masks)


class PhysioDataset(Dataset):
    def __init__(
        self, missing_ratio, is_train, observed_values, observed_masks, gt_masks
    ):
        self.missing_ratio = missing_ratio
        self.is_train = is_train
        idx2data = {}
        for idx in np.arange(len(observed_values)):
            if is_train:
                all_idxs = np.argwhere(observed_masks[idx])
                (all_times, all_feat_idxs) = (all_idxs[:, 0], all_idxs[:, 1])
                all_vals = observed_values[idx][all_times, all_feat_idxs]
                idx2data[idx] = {
                    "all_vals": all_vals,
                    "all_times": all_times,
                    "all_feat_idxs": all_feat_idxs,
                }

            else:
                obs_idxs = np.argwhere(gt_masks[idx])
                (obs_times, obs_feat_idxs) = (obs_idxs[:, 0], obs_idxs[:, 1])
                obs_vals = observed_values[idx][obs_times, obs_feat_idxs]

                unobs_mask = observed_masks[idx] - gt_masks[idx]
                unobs_idxs = np.argwhere(unobs_mask)
                (unobs_times, unobs_feat_idxs) = (unobs_idxs[:, 0], unobs_idxs[:, 1])
                unobs_vals = observed_values[idx][unobs_times, unobs_feat_idxs]

                idx2data[idx] = {
                    "obs_vals": obs_vals,
                    "obs_times": obs_times,
                    "obs_feat_idxs": obs_feat_idxs,
                    "unobs_vals": unobs_vals,
                    "unobs_times": unobs_times,
                    "unobs_feat_idxs": unobs_feat_idxs,
                }

        self.idx2data = idx2data

    def __getitem__(self, idx):
        if self.is_train:
            idx = np.random.randint(len(self.idx2data))
            data = self.idx2data[idx]

            all_vals = data["all_vals"]
            all_times = data["all_times"]
            all_feat_idxs = data["all_feat_idxs"]

            n = len(all_vals)
            all_idxs = np.arange(n)
            np.random.shuffle(all_idxs)
            n_missing = int(np.ceil(self.missing_ratio * n))
            obs_idxs = all_idxs[n_missing:]
            unobs_idxs = all_idxs[:n_missing]

            data = {
                "obs_vals": all_vals[obs_idxs],
                "obs_times": all_times[obs_idxs],
                "obs_feat_idxs": all_feat_idxs[obs_idxs],
                "unobs_vals": all_vals[unobs_idxs],
                "unobs_times": all_times[unobs_idxs],
                "unobs_feat_idxs": all_feat_idxs[unobs_idxs],
            }

        else:
            data = self.idx2data[idx]

        tensors = {}
        for (k, v) in data.items():
            T = torch.LongTensor if k.endswith("idxs") else torch.Tensor
            tensors[k] = T(v)

        return tensors

    def __len__(self):
        return len(self.idx2data)


def get_datasets(seed, missing_ratio):
    np.random.seed(seed)  # seed for ground truth choice

    try:
        observed_values = np.load("observed_values.npy")
        observed_masks = np.load("observed_masks.npy")
        gt_masks = np.load("gt_masks.npy")

    except FileNotFoundError:
        observed_values = []
        observed_masks = []
        gt_masks = []

        idlist = get_idlist()
        for id_ in idlist:
            try:
                id_data = parse_id(id_, missing_ratio)
                observed_values.append(id_data[0])
                observed_masks.append(id_data[1])
                gt_masks.append(id_data[2])
            except Exception as e:
                print(id_, e)
                continue

        observed_values = np.array(observed_values)
        observed_masks = np.array(observed_masks)
        gt_masks = np.array(gt_masks)

        # calc mean and std and normalize values
        # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
        tmp_values = observed_values.reshape(-1, 35)
        tmp_masks = observed_masks.reshape(-1, 35)
        mean = np.zeros(35)
        std = np.zeros(35)
        for k in range(35):
            c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
            mean[k] = c_data.mean()
            std[k] = c_data.std()

        observed_values = (observed_values - mean) / std * observed_masks

        np.save("observed_values.npy", observed_values)
        np.save("observed_masks.npy", observed_masks)
        np.save("gt_masks.npy", gt_masks)

    n_dataset = len(observed_values)
    n_feats = observed_values.shape[2]
    indlist = np.arange(n_dataset)
    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    test_n = int(0.2 * n_dataset)
    test_index = indlist[:test_n]
    remain_index = np.delete(indlist, np.arange(test_n))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    n_remain = len(remain_index)
    num_train = int(n_remain * 0.95)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    train_dataset = PhysioDataset(
        missing_ratio,
        True,
        observed_values[train_index],
        observed_masks[train_index],
        gt_masks[train_index],
    )
    valid_dataset = PhysioDataset(
        missing_ratio,
        False,
        observed_values[valid_index],
        observed_masks[valid_index],
        gt_masks[valid_index],
    )
    test_dataset = PhysioDataset(
        missing_ratio,
        False,
        observed_values[test_index],
        observed_masks[test_index],
        gt_masks[test_index],
    )
    return (train_dataset, valid_dataset, test_dataset, n_feats)
