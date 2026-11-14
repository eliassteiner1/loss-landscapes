import numpy as np
import shutil
import h5py

from torchvision import datasets

from loss_landscapes.paths import PROJ_ROOT




if __name__ == "__main__":
    print("\033[2J\033[3J\033[H", end="", flush=True) # start with an empty terminal
    print(f"\033[1m\033[38;2;51;153;102mrunning script {__file__}... \033[0m")
    
    # download data from torchvision fashonmnist
    print("⯈ generating data ...")
    ds_train = datasets.FashionMNIST(
        root = PROJ_ROOT / "data/tempdl",
        train = True,
        download = True,
    )

    ds_valid = datasets.FashionMNIST(
        root = PROJ_ROOT / "data/tempdl",
        train = False,
        download = True,
    )

    imgs = np.concat([ds_train.data.numpy(), ds_valid.data.numpy()], axis=0)
    imgs = imgs/255 # rescale values
    lbls = np.concat([ds_train.targets.numpy(), ds_valid.targets.numpy()], axis=0)

    # store data in a h5py file
    print("⯈ saving data ...")
    with h5py.File(PROJ_ROOT / "data/ds_fashion.h5", "w") as f:
        f.create_dataset("data", data=imgs)
        f.create_dataset("targets", data=lbls)
        
    print("⯈ done with data generation - file saved!")