import numpy as np
import sys
import time
import torch
import yaml

from deformer_tabular_ardm import DEformerTabularARDM
from settings import *
from torch import optim
from torch.utils.data import DataLoader
from uci_dataset import UCIDataset

SEED = 2010
torch.manual_seed(SEED)
torch.set_printoptions(linewidth=160)
np.random.seed(SEED)


def init_datasets(opts):
    dataset = opts["train"]["dataset"]
    batch_size = opts["train"]["batch_size"]

    train_dataset = UCIDataset(dataset, "train")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=opts["train"]["workers"],
    )
    valid_dataset = UCIDataset(dataset, "val")
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=opts["train"]["workers"],
    )
    test_dataset = UCIDataset(dataset, "test")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
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
    model_config["n_feats"] = train_dataset.n_feats
    model = DEformerTabularARDM(**model_config)
    return model


def get_loss(model, tensors, device):
    (pis, mus, sds) = model(tensors)
    vals = tensors["vals"].unsqueeze(2).to(device)
    comp_densities = (
        1 / (sds * np.sqrt(2 * np.pi)) * torch.exp(-1 / 2 * ((vals - mus) / sds) ** 2)
    )
    densities = (pis * comp_densities).sum(dim=2)
    loss = -torch.log(densities).mean()
    return loss


def train_model():
    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    optimizer = optim.Adam(train_params, lr=opts["train"]["learning_rate"])

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
            for (batch_idx, valid_tensors) in enumerate(valid_loader):
                loss = get_loss(model, valid_tensors, device)
                total_valid_loss += loss.item()
                n_valid += 1

            total_valid_loss /= n_valid

        if total_valid_loss < best_valid_loss:
            best_valid_loss = total_valid_loss
            no_improvement = 0
            torch.save(optimizer.state_dict(), f"{JOB_DIR}/optimizer.pth")
            torch.save(model.state_dict(), f"{JOB_DIR}/best_params.pth")

            test_loss_best_valid = 0.0
            with torch.no_grad():
                n_test = 0
                for (batch_idx, test_tensors) in enumerate(test_loader):
                    loss = get_loss(model, test_tensors, device)
                    test_loss_best_valid += loss.item()
                    n_test += 1

            test_loss_best_valid /= n_test

        elif no_improvement < opts["train"]["patience"]:
            no_improvement += 1
            if no_improvement == opts["train"]["patience"]:
                print("Reducing learning rate.")
                no_improvement = 0
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
        for (batch_idx, train_tensors) in enumerate(train_loader):
            if batch_idx % 1000 == 0:
                print(batch_idx, flush=True)

            optimizer.zero_grad()
            loss = get_loss(model, train_tensors, device)
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
    mix_comps = opts["model"]["mix_comps"]

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
