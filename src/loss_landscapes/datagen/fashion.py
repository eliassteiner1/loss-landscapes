from torchvision import datasets
import numpy as np
from loss_landscapes.paths import PROJ_ROOT



def generate_fashionmnist_data() -> tuple[np.ndarray, np.ndarray]:
    
    # download data from torchvision fashonmnist
    print("> donwloading fashionMnist from torchvision")
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

    print("> creating fashion dataset ...")
    data = np.concat([ds_train.data.numpy(), ds_valid.data.numpy()], axis=0)
    data = data/255 # rescale values
    targets = np.concat([ds_train.targets.numpy(), ds_valid.targets.numpy()], axis=0)

    return data, targets
    
