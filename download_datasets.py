import numpy as np
import os
import pickle
import shutil
import urllib.request

from PIL import Image


def download_cifar10():
    print("Downloading CIFAR-10...")
    url_prefix = "https://www.cs.toronto.edu/~kriz"
    filename = "cifar-10-python.tar.gz"
    url = f"{url_prefix}/{filename}"
    print(f"Downloading from: {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Saved to {filename}.")

    shutil.unpack_archive(filename)

    data = []
    for batch in range(1, 6):
        with open(f"cifar-10-batches-py/data_batch_{batch}", "rb") as fo:
            data_dict = pickle.load(fo, encoding="bytes")
            data.append(data_dict[b"data"])

    data = np.concatenate(data)
    np.save("cifar10_train.npy", data)

    with open("cifar-10-batches-py/test_batch", "rb") as fo:
        data_dict = pickle.load(fo, encoding="bytes")
        np.save("cifar10_test.npy", data_dict[b"data"])


def download_mnist():
    # Adapted from: https://github.com/yoonholee/pytorch-vae/blob/master/data_loader/fixed_mnist.py.
    url_prefix = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist"
    print("Downloading binarized MNIST...")
    data = {}
    for dataset in ["train", "valid", "test"]:
        filename = f"binarized_mnist_{dataset}.amat"
        url = f"{url_prefix}/{filename}"
        print(f"Downloading from: {url}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Saved to {filename}.")
        with open(filename) as f:
            arr = []
            for line in f:
                arr.append([int(pix) for pix in line.split()])

            arr = np.array(arr, dtype="int8").reshape(-1, 28, 28)
            # To plot one use: Image.fromarray(255 * arr[idx]).show().
            data[dataset] = arr

    data["train"] = np.concatenate([data["train"], data["valid"]])
    np.save("mnist_train.npy", data["train"])
    np.save("mnist_test.npy", data["test"])


def save_some_mnist():
    mnist_train = np.array(np.load("mnist_train.npy"), dtype="uint8")
    idxs = np.arange(len(mnist_train))

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/results", exist_ok=True)

    samples = 20
    for idx in idxs[:samples]:
        img = Image.fromarray(255 * mnist_train[idx])
        img.save(f"{home_dir}/results/{idx}.jpg")

    shutil.make_archive(f"{home_dir}/results", "zip", f"{home_dir}/results")
    shutil.rmtree(f"{home_dir}/results")
