import torch
import torch.nn as nn
import numpy as np
import tyro
from typing import Literal

from loss_landscapes.paths import PROJ_ROOT
from loss_landscapes.inference.evaluate import generate_random_dir_naive
from loss_landscapes.inference.evaluate import generate_random_dir_filternorm
from loss_landscapes.inference.evaluate import evaluate_grid
from loss_landscapes.networks.io import load_model
from loss_landscapes.train.setup import get_device
from loss_landscapes.datagen.io import load_data


def main(
    checkpoint: str, 
    method: Literal["naive", "filternorm"] = "filternorm", 
    reach: float = 1.0, 
    npoints: int = 51,
):
    
    # load model from snapshot
    device = get_device()
    network, data_mean, data_sdev = load_model(name=checkpoint)
    network.to(device).eval()
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    
    # load the data
    data, targets = load_data("ds_fashion.h5")
    data = (data - data_mean) / data_sdev # normalize data like how it was during training
    data = torch.as_tensor(data, dtype=torch.float32)[:, None, :, :].to(device) # add 1 channel singlt. dim for conv
    targets = torch.as_tensor(targets, dtype=torch.long).to(device) # nn.CrossEntropyLoss needs int
    
    # 3. generate the two random directions in param space, starting from the model weights
    trained_params = {k: v.detach().clone() for k, v in network.named_parameters()} # copy params from model on device!
    
    if method == "naive":
        dir1 = generate_random_dir_naive(trained_params)
        dir2 = generate_random_dir_naive(trained_params)
    elif method == "filternorm":
        dir1 = generate_random_dir_filternorm(trained_params)
        dir2 = generate_random_dir_filternorm(trained_params)  
    else:
        raise(ValueError(f"choose method from [naive, filternorm]! (got {method})"))
    
    # evaluate a grid of model weights
    alphas, betas, z = evaluate_grid(
        data, targets,
        network,
        trained_params,
        device,
        loss_function,
        dir1,
        dir2,
        reach = reach,
        npoints = npoints,
    )
                
    # store results for plotting
    print(f"> saving result ... ")
    np.savez(
        file=PROJ_ROOT / "results" / f"{checkpoint.replace(".pt", "")}_{method}.npz",
        z=z, alphas=alphas, betas=betas,
    )
    print("> done!")


if __name__ == "__main__":
    tyro.cli(main)