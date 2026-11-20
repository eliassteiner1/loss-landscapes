import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call
from tqdm.auto import tqdm

from loss_landscapes.types import LossFunction, Params

def _loss_at(
    network: nn.Module, 
    params: Params, 
    data: torch.Tensor, 
    targets: torch.Tensor, 
    loss_function: LossFunction,
) -> torch.Tensor:
    
    """ to evaluate a model with data AND parameters """
    pred = functional_call(network, params, (data, ))
    loss = loss_function(pred, targets)
    return loss


def _filternorm(weights: torch.Tensor) -> torch.Tensor:
    """ one norm per output filter/neuron: norm over all dims except dim 0 """
    norms = weights.flatten(1).norm(dim=1) # shape (out,)
    return norms.view(-1, *[1] * (weights.dim() - 1)) # (out, 1, 1, 1) or (out, 1), so it broadcasts


def generate_random_dir_naive(trained_params: Params) -> Params:
    """ global normalization: the direction as a whole (all layers concatenated)
    gets the same total norm as the trained weights, no per-layer balancing """

    direction = {}
    for name, param in trained_params.items():

        if param.dim() > 1:
            # totally random initialization on weights
            direction[name] = torch.randn_like(param)
        else:
            # biases and batch norm should stay at their trained values
            direction[name] = torch.zeros_like(param)

    # total norms over all perturbed params (the norm of everything flattened into one long vector)
    perturbed = [k for k, w in trained_params.items() if w.dim() > 1]
    dir_norm = torch.sqrt(sum((direction[k]**2).sum() for k in perturbed))
    trained_norm = torch.sqrt(sum(trained_params[k].pow(2).sum() for k in perturbed))

    # one single scale factor for the whole network
    scale = trained_norm / dir_norm
    for k in perturbed:
        direction[k] *= scale

    return direction


def generate_random_dir_filternorm(trained_params: Params) -> Params:
    " proper filter normalized directions as in [https://arxiv.org/abs/1712.09913]"

    direction = {}
    for name, params in trained_params.items():

        if params.dim() > 1:

            randn_dir = torch.randn_like(params)
            normed_dir = randn_dir / _filternorm(randn_dir)
            scaled_dir = normed_dir * _filternorm(params)
            direction[name] = scaled_dir

        else:

            direction[name] = torch.zeros_like(params)

    return direction


def evaluate_grid(
    data: torch.Tensor, targets: torch.Tensor,
    network: nn.Module, 
    trained_params: Params,
    device: str,
    loss_function: LossFunction,
    dir1: Params,
    dir2: Params,
    reach: float = 1.0,
    npoints: int = 51,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    alphas = np.linspace(-reach, reach, npoints)
    betas  = np.linspace(-reach, reach, npoints)

    z = torch.zeros(npoints, npoints, device=device)

    # double loop iteration (w\ torch no grad)
    print("> evaluating model on parameter grid ...")
    with torch.no_grad():
        for i, a in tqdm(enumerate(alphas), desc="  progress:",  total=npoints, disable=None):
            for j, b in enumerate(betas):
                perturbed_params = {k: trained_params[k] + a*dir1[k] + b*dir2[k] for k in trained_params}
                z[i, j] = _loss_at(network, perturbed_params, data[:5000], targets[:5000], loss_function)

    return alphas, betas, z.cpu().numpy()