import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import numpy as np
from tqdm.auto import tqdm
import h5py

from loss_landscapes.paths import PROJ_ROOT


    
if __name__ == "__main__":
    print("\033[2J\033[3J\033[H", end="", flush=True) # start with an empty terminal
    print(f"\033[1m\033[38;2;51;153;102mrunning script {__file__}... \033[0m")
    
    # Device setup 
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"⯈ using {DEVICE} as device")
    
    # setup training
    print("⯈ loading dataset ...")
    with h5py.File(PROJ_ROOT / "data/dataset_2dfunc.h5", "r") as f:
        ...
    
    
