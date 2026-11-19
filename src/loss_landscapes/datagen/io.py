

import h5py
import numpy as np


from pathlib import Path


def load_data(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    print(f"> loading dataset {dataset_path} ...")
    with h5py.File(dataset_path, "r") as f:
        data = f["data"][...]
        targets = f["targets"][...]
    print("> successfully loaded data!")
    return data, targets

def write_data(dataset_path: Path, data: np.ndarray, targets: np.ndarray):
    print(f"> writing dataset {dataset_path}")
    with h5py.File(dataset_path, "w") as f:
        f["data"] = data
        f["targets"] = targets
