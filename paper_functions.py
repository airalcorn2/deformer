import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

from binarized_notmnist_dataset import NotMNISTDataset
from PIL import Image
from settings import *
from torch import nn
from torch.utils.data import DataLoader
from train_deformer import init_datasets, init_model


def get_multi_order_nlls():
    JOB = "20210514165425"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))

    # Initialize datasets.
    (train_dataset, _, _, _, _, test_loader) = init_datasets(opts)

    # Initialize model.
    device = torch.device("cuda:0")
    model = init_model(opts, train_dataset).to(device)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    orders = 10
    n_pix = 28 ** 2
    pix_idxs = np.arange(n_pix)
    np.random.seed(2010)
    test_nlls = []
    for order in range(orders):
        print(order, flush=True)
        np.random.shuffle(pix_idxs)
        with torch.no_grad():
            for (test_idx, test_tensors) in enumerate(test_loader):
                test_tensors["pixels"] = test_tensors["pixels"][pix_idxs]
                test_tensors["positions"] = test_tensors["positions"][pix_idxs]
                preds = model(test_tensors).flatten()
                labels = test_tensors["pixels"].flatten().to(device)
                loss = criterion(preds, labels)
                test_nlls.append(n_pix * loss.item())

    print(sum(test_nlls) / len(test_nlls))
    fg = sns.displot(test_nlls)
    plt.xlabel("NLL")
    plt.tight_layout()
    fg.fig.savefig("/home/michael/test_nlls_hist.png")

    notmnist_dataset = NotMNISTDataset()
    notmnist_loader = DataLoader(
        dataset=notmnist_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )
    test_idxs = set(
        np.random.choice(
            np.arange(len(notmnist_dataset)), len(test_nlls) // orders, False
        )
    )
    pix_idxs = np.arange(n_pix)
    np.random.seed(2010)
    nlls = []
    for order in range(orders):
        print(order, flush=True)
        np.random.shuffle(pix_idxs)
        with torch.no_grad():
            for (test_idx, test_tensors) in enumerate(notmnist_loader):
                if test_idx not in test_idxs:
                    continue

                test_tensors = notmnist_dataset[test_idx]
                test_tensors["pixels"] = test_tensors["pixels"][pix_idxs]
                test_tensors["positions"] = test_tensors["positions"][pix_idxs]
                preds = model(test_tensors).flatten()
                labels = test_tensors["pixels"].flatten().to(device)
                loss = criterion(preds, labels)
                nlls.append(n_pix * loss.item())

    fg = sns.displot(nlls)
    plt.xlabel("NLL")
    plt.tight_layout()
    fg.fig.savefig("/home/michael/not_nlls_hist.png")

    df = pd.DataFrame(
        {
            "NLL": test_nlls + nlls,
            "dataset": len(test_nlls) * ["MNIST"] + len(nlls) * ["notMNIST"],
        }
    )
    fg = sns.displot(
        data=df,
        x="NLL",
        hue="dataset",
        kind="kde",
        fill=True,
        palette=sns.color_palette("bright")[:2],
        height=5,
        aspect=1.5,
    )
    plt.xlabel("NLL")
    plt.tight_layout()
    fg.fig.savefig("/home/michael/all_nlls_hist.png")


def get_test_imgs_nll():
    JOB = "20210514165425"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))

    # Initialize datasets.
    (train_dataset, _, _, _, test_dataset, _) = init_datasets(opts)

    # Initialize model.
    device = torch.device("cuda:0")
    model = init_model(opts, train_dataset).to(device)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    rows = 5
    cols = 10
    samples = rows * cols
    idxs = np.random.choice(np.arange(len(test_dataset)), samples, False)

    img_size = train_dataset.img_size
    n_pix = 28 ** 2
    pix_idxs = np.arange(n_pix)

    orders = 10
    criterion = nn.BCEWithLogitsLoss()
    nlls = []
    for idx in idxs:
        tensors = test_dataset[idx]
        nll = 0
        for order in range(orders):
            np.random.shuffle(pix_idxs)
            tensors["pixels"] = tensors["pixels"][pix_idxs]
            tensors["positions"] = tensors["positions"][pix_idxs]
            with torch.no_grad():
                preds = model(tensors).flatten()
                labels = tensors["pixels"].flatten().to(device)
                nll += img_size ** 2 * criterion(preds, labels).item()

        nlls.append(nll / orders)

    sorted_idxs = np.argsort(nlls)
    print(nlls[sorted_idxs[0]])
    print(nlls[sorted_idxs[-1]])
    img_arr = np.zeros((rows * (img_size + 1), cols * (img_size + 1)), dtype="uint8")
    for (img_idx, idx) in enumerate(sorted_idxs):
        row = (img_size + 1) * (img_idx // cols)
        col = (img_size + 1) * (img_idx % cols)
        img_arr[row : row + img_size, col : col + img_size] = test_dataset.img_data[
            idxs[img_idx]
        ]

    img_arr[img_size :: img_size + 1, :] = 1
    img_arr[:, img_size :: img_size + 1] = 1
    home_dir = os.path.expanduser("~")
    img = Image.fromarray(255 * img_arr[:-1, :-1])
    img.save(f"{home_dir}/test_nlls.jpg")


def generate_samples():
    JOB = "20210514165425"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))

    # Initialize datasets.
    (train_dataset, _, _, _, _, _) = init_datasets(opts)

    # Initialize model.
    device = torch.device("cuda:0")
    model = init_model(opts, train_dataset).to(device)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    rows = 5
    cols = 10
    img_size = train_dataset.img_size

    orders = 10
    n_pix = 28 ** 2
    pix_idxs = np.arange(n_pix)
    nlls = []
    img_arrs = []
    criterion = nn.BCEWithLogitsLoss()
    for row in range(rows):
        for col in range(cols):
            print(col, flush=True)
            tensors = train_dataset[0]
            img_arr = np.zeros((img_size, img_size), dtype="uint8")
            np.random.shuffle(pix_idxs)
            tensors["pixels"] = tensors["pixels"][pix_idxs]
            tensors["positions"] = tensors["positions"][pix_idxs]

            for (pix_idx, pix_pos) in enumerate(tensors["positions"]):
                with torch.no_grad():
                    preds = model(tensors)

                tensors["pixels"][pix_idx] = torch.bernoulli(
                    torch.sigmoid(preds[pix_idx])
                )
                (row, col) = pix_pos.int().detach().cpu().numpy()
                img_arr[row, col] = tensors["pixels"][pix_idx].int().item()

            nll = 0
            for order in range(orders):
                np.random.shuffle(pix_idxs)
                tensors["pixels"] = tensors["pixels"][pix_idxs]
                tensors["positions"] = tensors["positions"][pix_idxs]
                with torch.no_grad():
                    preds = model(tensors).flatten()
                    labels = tensors["pixels"].flatten().to(device)
                    nll += img_size ** 2 * criterion(preds, labels).item()

            nlls.append(nll / orders)
            img_arrs.append(img_arr)

    sorted_idxs = np.argsort(nlls)
    print(nlls[sorted_idxs[0]])
    print(nlls[sorted_idxs[-1]])
    img_arr = np.zeros((rows * (img_size + 1), cols * (img_size + 1)), dtype="uint8")
    for (img_idx, idx) in enumerate(sorted_idxs):
        row = (img_size + 1) * (img_idx // cols)
        col = (img_size + 1) * (img_idx % cols)
        img_arr[row : row + img_size, col : col + img_size] = img_arrs[idx]

    img_arr[img_size :: img_size + 1, :] = 1
    img_arr[:, img_size :: img_size + 1] = 1
    home_dir = os.path.expanduser("~")
    img = Image.fromarray(255 * img_arr[:-1, :-1])
    img.save(f"{home_dir}/gen.jpg")


def permute_fill_tensors(tensors, skip_pix):
    cond_pix = len(tensors["pixels"]) - skip_pix
    cond_idxs = torch.randperm(cond_pix)
    pred_idxs = cond_pix + torch.randperm(skip_pix)
    tensors["pixels"][:cond_pix] = tensors["pixels"][cond_idxs]
    tensors["positions"][:cond_pix] = tensors["positions"][cond_idxs]
    tensors["pixels"][cond_pix:] = tensors["pixels"][pred_idxs]
    tensors["positions"][cond_pix:] = tensors["positions"][pred_idxs]
    return tensors


def get_square_skips(img_size, sq_size, test_idx, test_dataset):
    skip_row = img_size // 2 - sq_size // 2
    skip_col = img_size // 2 - sq_size // 2
    tensors = {"pixels": [], "positions": []}
    for row in range(img_size):
        for col in range(img_size):
            if (skip_row <= row < skip_row + sq_size) and (
                skip_col <= col < skip_col + sq_size
            ):
                continue
            else:
                tensors["pixels"].append(test_dataset.img_data[test_idx, row, col])
                tensors["positions"].append((row, col))

    for row in range(skip_row, skip_row + sq_size):
        for col in range(skip_col, skip_col + sq_size):
            tensors["pixels"].append(test_dataset.img_data[test_idx, row, col])
            tensors["positions"].append((row, col))

    tensors["pixels"] = torch.Tensor(tensors["pixels"])
    tensors["positions"] = torch.Tensor(tensors["positions"])
    return permute_fill_tensors(tensors, sq_size ** 2)


def get_row_skips(
    full_skip_rows, last_skip_row, skip_cols, img_size, test_idx, test_dataset
):
    tensors = {"pixels": [], "positions": []}

    for row in range(img_size):
        if row in full_skip_rows:
            continue

        for col in range(img_size):
            if (row == last_skip_row) and (col in skip_cols):
                continue
            else:
                tensors["pixels"].append(test_dataset.img_data[test_idx, row, col])
                tensors["positions"].append((row, col))

    for row in full_skip_rows:
        for col in range(img_size):
            tensors["pixels"].append(test_dataset.img_data[test_idx, row, col])
            tensors["positions"].append((row, col))

    for col in skip_cols:
        tensors["pixels"].append(test_dataset.img_data[test_idx, last_skip_row, col])
        tensors["positions"].append((last_skip_row, col))

    tensors["pixels"] = torch.Tensor(tensors["pixels"])
    tensors["positions"] = torch.Tensor(tensors["positions"])
    return permute_fill_tensors(
        tensors, len(full_skip_rows) * img_size + len(skip_cols)
    )


def get_random_skips(skip_idxs, img_size, test_idx, test_dataset):
    tensors = {"pixels": [], "positions": []}

    for idx in range(img_size ** 2):
        if idx in skip_idxs:
            continue

        row = idx // img_size
        col = idx % img_size
        tensors["pixels"].append(test_dataset.img_data[test_idx, row, col])
        tensors["positions"].append((row, col))

    for idx in skip_idxs:
        row = idx // img_size
        col = idx % img_size
        tensors["pixels"].append(test_dataset.img_data[test_idx, row, col])
        tensors["positions"].append((row, col))

    tensors["pixels"] = torch.Tensor(tensors["pixels"])
    tensors["positions"] = torch.Tensor(tensors["positions"])
    return permute_fill_tensors(tensors, len(skip_idxs))


def complete_samples():
    JOB = "20210514165425"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))

    # Initialize datasets.
    (train_dataset, _, _, _, test_dataset, _) = init_datasets(opts)

    # Initialize model.
    device = torch.device("cuda:0")
    model = init_model(opts, train_dataset).to(device)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    img_size = train_dataset.img_size
    sq_size = 10
    full_skip_rows = [5, 10, 15]
    last_skip_row = 20
    skip_cols = np.arange(16) + 6
    n_skip = sq_size ** 2
    skip_idxs = np.random.choice(np.arange(img_size ** 2), n_skip, False)
    missing_types = ["square", "row", "random"]

    test_idxs = [2501, 3501, 4501, 8501, 9501]
    rows = len(test_idxs)
    cols = 1 + 2 * len(missing_types)
    img_arr = np.zeros((rows * (img_size + 1), cols * (img_size + 1), 3), dtype="uint8")

    for (row_idx, test_idx) in enumerate(test_idxs):
        row = row_idx * (img_size + 1)
        img_arr[row : row + img_size, :img_size] = test_dataset.img_data[test_idx][
            ..., np.newaxis
        ]

        for (missing_idx, missing_type) in enumerate(missing_types):
            if missing_type == "square":
                tensors = get_square_skips(img_size, sq_size, test_idx, test_dataset)
            elif missing_type == "row":
                tensors = get_row_skips(
                    full_skip_rows,
                    last_skip_row,
                    skip_cols,
                    img_size,
                    test_idx,
                    test_dataset,
                )
            else:
                tensors = get_random_skips(skip_idxs, img_size, test_idx, test_dataset)

            filled_arr = np.zeros((img_size, img_size), dtype="uint8")
            missing_arr = np.zeros((img_size, img_size, 3), dtype="uint8")
            for (pix_idx, pix_pos) in enumerate(tensors["positions"]):
                (row, col) = pix_pos.int().detach().cpu().numpy()
                if pix_idx < img_size ** 2 - sq_size ** 2:
                    pix_val = tensors["pixels"][pix_idx].int().item()
                    filled_arr[row, col] = pix_val
                    missing_arr[row, col] = pix_val
                    continue
                else:
                    missing_arr[row, col, 0] = 1

                with torch.no_grad():
                    preds = model(tensors)

                tensors["pixels"][pix_idx] = torch.bernoulli(
                    torch.sigmoid(preds[pix_idx])
                )
                filled_arr[row, col] = tensors["pixels"][pix_idx].int().item()

            row = row_idx * (img_size + 1)
            col = (img_size + 1) + 2 * missing_idx * (img_size + 1)
            img_arr[row : row + img_size, col : col + img_size] = missing_arr
            col += img_size + 1
            img_arr[row : row + img_size, col : col + img_size] = filled_arr[
                ..., np.newaxis
            ]

    img_arr[img_size :: img_size + 1, :] = 1
    img_arr[:, img_size :: img_size + 1] = 1
    img = Image.fromarray(255 * img_arr[:-1, :-1])
    home_dir = os.path.expanduser("~")
    img.save(f"{home_dir}/filled.jpg")
