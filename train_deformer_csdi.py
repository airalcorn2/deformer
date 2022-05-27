import numpy as np
import time
import torch

from csdi_physio_dataset import get_datasets
from deformer_csdi import DEformerCSDI
from torch import nn, optim
from torch.utils.data import DataLoader

SEED = 2010
torch.manual_seed(SEED)
torch.set_printoptions(linewidth=160)
np.random.seed(SEED)


def worker_init_fn(worker_id):
    # NumPy seed takes a 32-bit unsigned integer.
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32))


def init_model(opts):
    model_config = opts["model"]
    model = DEformerCSDI(**model_config)
    return model


def get_loss(model, tensors, device):
    preds = model(tensors).flatten()
    vals = tensors["unobs_vals"].to(device)
    loss = criterion(preds, vals)
    return loss


def train_model():
    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    optimizer = optim.Adam(train_params, lr=opts["train"]["learning_rate"])

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
                n_unobs = len(valid_tensors["unobs_vals"])
                total_valid_loss += loss.item() * n_unobs
                n_valid += n_unobs

            total_valid_loss /= n_valid

        if total_valid_loss < best_valid_loss:
            best_valid_loss = total_valid_loss
            no_improvement = 0
            torch.save(model.state_dict(), "csdi_best_params.pth")

            test_loss_best_valid = 0.0
            with torch.no_grad():
                n_test = 0
                for (batch_idx, test_tensors) in enumerate(test_loader):
                    loss = get_loss(model, test_tensors, device)
                    n_unobs = len(test_tensors["unobs_vals"])
                    test_loss_best_valid += loss.item() * n_unobs
                    n_test += n_unobs

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
            n_unobs = len(train_tensors["unobs_vals"])
            loss = get_loss(model, train_tensors, device)
            if torch.isnan(loss):
                raise ValueError

            total_train_loss += loss.item() * n_unobs
            loss.backward()
            optimizer.step()
            n_train += n_unobs

        epoch_time = time.time() - start_time

        total_train_loss /= n_train
        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss

        print(f"epoch_time: {epoch_time:.2f}", flush=True)


if __name__ == "__main__":
    opts = {
        "train": {
            "missing_ratio": 0.1,
            "workers": 10,
            "learning_rate": 1.0e-5,
            "patience": 5,
        },
        "model": {
            "idx_embed_dim": 20,
            "mlp_layers": [128, 256, 512],
            "nhead": 8,
            "dim_feedforward": 2048,
            "num_layers": 6,
            "dropout": 0.0,
        },
    }

    # Initialize datasets.
    (train_dataset, valid_dataset, test_dataset, n_feats) = get_datasets(
        1, opts["train"]["missing_ratio"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        worker_init_fn=worker_init_fn,
        num_workers=opts["train"]["workers"],
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None, num_workers=opts["train"]["workers"]
    )
    test_loader = DataLoader(
        test_dataset, batch_size=None, num_workers=opts["train"]["workers"]
    )

    # Initialize model.
    device = torch.device("cuda:0")
    opts["model"]["n_feats"] = n_feats
    model = init_model(opts).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")

    criterion = nn.L1Loss()

    train_model()
