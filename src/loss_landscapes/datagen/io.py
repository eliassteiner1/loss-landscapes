

import h5py
import numpy as np


from pathlib import Path

from loss_landscapes.paths import PROJ_ROOT


def load_data(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    print(f"> loading dataset {dataset_name} ...")
    with h5py.File(PROJ_ROOT / "data" / dataset_name, "r") as f:
        data = f["data"][...]
        targets = f["targets"][...]
    print("> successfully loaded data!")
    return data, targets

def write_data(dataset_name: str, data: np.ndarray, targets: np.ndarray) -> None:
    print(f"> writing dataset {dataset_name}")
    with h5py.File(PROJ_ROOT / "data" / dataset_name, "w") as f:
        f["data"] = data
        f["targets"] = targets
