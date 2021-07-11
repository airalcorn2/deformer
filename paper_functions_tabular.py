import numpy as np
import torch
import yaml

from settings import *
from train_deformer_tabular import get_loss, init_datasets, init_model


def get_multi_order_nlls():
    JOB = "20210705054621"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))

    # Initialize datasets.
    (train_dataset, _, _, _, _, test_loader) = init_datasets(opts)

    # Initialize model.
    device = torch.device("cuda:0")
    model = init_model(opts, train_dataset).to(device)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    orders = 10
    n_feats = train_dataset.n_feats
    feat_idxs = np.arange(train_dataset.n_feats)
    np.random.seed(2010)
    test_nlls = []
    for order in range(orders):
        print(order, flush=True)
        np.random.shuffle(feat_idxs)
        with torch.no_grad():
            for (test_idx, test_tensors) in enumerate(test_loader):
                test_tensors["idxs"] = test_tensors["idxs"][:, feat_idxs]
                test_tensors["vals"] = test_tensors["vals"][:, feat_idxs]
                loss = get_loss(model, test_tensors, device)
                test_nlls.append(n_feats * loss.item())

    print(sum(test_nlls) / len(test_nlls))
