import numpy as np
import sys
import time
import torch
import yaml

from deformer import DEformer
from deformer_dataset import DEformerDataset
from download_datasets import download_cifar10, download_mnist
from settings import *
from torch import nn, optim
from torch.utils.data import DataLoader

SEED = 2010
torch.manual_seed(SEED)
torch.set_printoptions(linewidth=160)
np.random.seed(SEED)


def init_datasets(opts):
    dataset = opts["train"]["dataset"]

    try:
        train_data = np.load(f"{dataset}_train.npy")
        test_data = np.load(f"{dataset}_test.npy")
    except FileNotFoundError:
        eval(f"download_{dataset}()")
        train_data = np.load(f"{dataset}_train.npy")
        test_data = np.load(f"{dataset}_test.npy")

    train_valid_idxs = np.arange(len(train_data))
    np.random.shuffle(train_valid_idxs)
    n_train = int(opts["train"]["train_prop"] * len(train_valid_idxs))
    train_idxs = train_valid_idxs[:n_train]
    valid_idxs = train_valid_idxs[n_train:]

    train_dataset = DEformerDataset(dataset, train_data[train_idxs], True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=opts["train"]["workers"],
    )
    valid_dataset = DEformerDataset(dataset, train_data[valid_idxs], False)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )
    test_dataset = DEformerDataset(dataset, test_data, False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )

    return (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    )


def init_model(opts, train_dataset):
    model_config = opts["model"]
    model_config["img_size"] = train_dataset.img_size
    model_config["pos_in_feats"] = 2 if opts["train"]["dataset"] == "mnist" else 3
    model_config["pixel_n"] = 1 if opts["train"]["dataset"] == "mnist" else 256
    model = DEformer(**model_config)
    return model


def get_preds_labels(tensors):
    preds = model(tensors)
    labels = tensors["pixels"].flatten().to(device)
    if dataset == "mnist":
        preds = preds.flatten()
    else:
        labels = labels.long()

    return (preds, labels)


def train_model():
    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    optimizer = optim.Adam(train_params, lr=opts["train"]["learning_rate"])
    if dataset == "mnist":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Continue training on a prematurely terminated model.
    try:
        model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))

        try:
            state_dict = torch.load(f"{JOB_DIR}/optimizer.pth")
            if opts["train"]["learning_rate"] == state_dict["param_groups"][0]["lr"]:
                optimizer.load_state_dict(state_dict)

        except ValueError:
            print("Old optimizer doesn't match.")

    except FileNotFoundError:
        pass

    best_train_loss = float("inf")
    best_valid_loss = float("inf")
    test_loss_best_valid = float("inf")
    total_train_loss = None
    no_improvement = 0
    for epoch in range(1000000):
        print(f"\nepoch: {epoch}", flush=True)

        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            n_valid = 0
            for (valid_idx, valid_tensors) in enumerate(valid_loader):
                (preds, labels) = get_preds_labels(valid_tensors)
                loss = criterion(preds, labels)
                total_valid_loss += loss.item()
                n_valid += 1

            if dataset == "mnist":
                probs = 1 / (1 + (-preds).exp())
                preds = (probs > 0.5).int()

            else:
                probs = torch.softmax(preds, dim=1)
                (probs, preds) = probs.max(1)

            print(probs)
            print(preds)
            print(labels.int(), flush=True)

            total_valid_loss /= n_valid

        if total_valid_loss < best_valid_loss:
            best_valid_loss = total_valid_loss
            no_improvement = 0
            torch.save(optimizer.state_dict(), f"{JOB_DIR}/optimizer.pth")
            torch.save(model.state_dict(), f"{JOB_DIR}/best_params.pth")

            test_loss_best_valid = 0.0
            with torch.no_grad():
                n_test = 0
                for (test_idx, test_tensors) in enumerate(test_loader):
                    (preds, labels) = get_preds_labels(test_tensors)
                    loss = criterion(preds, labels)
                    test_loss_best_valid += loss.item()
                    n_test += 1

            test_loss_best_valid /= n_test

        elif no_improvement < opts["train"]["patience"]:
            no_improvement += 1
            if no_improvement == opts["train"]["patience"]:
                print("Reducing learning rate.")
                for g in optimizer.param_groups:
                    g["lr"] *= 0.1

        print(f"total_train_loss: {total_train_loss}")
        print(f"best_train_loss: {best_train_loss}")
        print(f"total_valid_loss: {total_valid_loss}")
        print(f"best_valid_loss: {best_valid_loss}")
        print(f"test_loss_best_valid: {test_loss_best_valid}")

        model.train()
        total_train_loss = 0.0
        n_train = 0
        start_time = time.time()
        for (train_idx, train_tensors) in enumerate(train_loader):
            if train_idx % 1000 == 0:
                print(train_idx, flush=True)

            optimizer.zero_grad()
            (preds, labels) = get_preds_labels(train_tensors)
            loss = criterion(preds, labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1

        epoch_time = time.time() - start_time

        total_train_loss /= n_train
        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss

        print(f"epoch_time: {epoch_time:.2f}", flush=True)


if __name__ == "__main__":
    JOB = sys.argv[1]
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    except IndexError:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    dataset = opts["train"]["dataset"]
    assert dataset in {"mnist", "cifar10"}

    # Initialize datasets.
    (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    ) = init_datasets(opts)

    # Initialize model.
    device = torch.device("cuda:0")
    model = init_model(opts, train_dataset).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")

    train_model()
