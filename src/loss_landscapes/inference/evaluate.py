import numpy as np
import torch
from torch.func import functional_call
from tqdm.auto import tqdm


def _loss_at(model, model_params, X: torch.Tensor, y: torch.Tensor, lossF) -> torch.Tensor:
    """ to evaluate a model with data AND parameters """
    pred = functional_call(model, model_params, (X, ))
    loss = lossF(pred, y)
    return loss


def _filternorm(weights):
    """ one norm per output filter/neuron: norm over all dims except dim 0 """
    norms = weights.flatten(1).norm(dim=1) # shape (out,)
    return norms.view(-1, *[1] * (weights.dim() - 1)) # (out, 1, 1, 1) or (out, 1), so it broadcasts


def generate_random_dir_naive(model_params):
    """ global normalization: the direction as a whole (all layers concatenated)
    gets the same total norm as the trained weights, no per-layer balancing """

    param_dir = {}
    for layer_name, weights in model_params.items():

        if weights.dim() > 1:
            # totally random initialization on weights
            param_dir[layer_name] = torch.randn_like(weights)
        else:
            # biases and batch norm should stay at their trained values
            param_dir[layer_name] = torch.zeros_like(weights)

    # total norms over all perturbed params (the norm of everything flattened into one long vector)
    perturbed = [k for k, w in model_params.items() if w.dim() > 1]
    dir_norm = torch.sqrt(sum((param_dir[k]**2).sum() for k in perturbed))
    theta_norm = torch.sqrt(sum(model_params[k].pow(2).sum() for k in perturbed))

    # one single scale factor for the whole network
    scale = theta_norm / dir_norm
    for k in perturbed:
        param_dir[k] *= scale

    return param_dir


def generate_random_dir_filternorm(model_params):
    " proper filter normalized directions as in [https://arxiv.org/abs/1712.09913]"

    param_dir = {}
    for layer_name, weights in model_params.items():

        if weights.dim() > 1:

            randn_dir = torch.randn_like(weights)
            normed_dir = randn_dir / _filternorm(randn_dir)
            scaled_dir = normed_dir * _filternorm(weights)
            param_dir[layer_name] = scaled_dir

        else:

            param_dir[layer_name] = torch.zeros_like(weights)

    return param_dir


def evaluate_grid(
    data, targets,
    model,
    trained_weights,
    device,
    lossF,
    dir1,
    dir2,

    reach = 1.0,
    npoints = 51,
    ):

    alphas = np.linspace(-reach, reach, npoints)
    betas  = np.linspace(-reach, reach, npoints)

    z = torch.zeros(npoints, npoints, device=device)

    # double loop iteration (w\ torch no grad)
    print("> evaluating model on parameter grid ...")
    with torch.no_grad():
        for i, a in tqdm(enumerate(alphas), desc="  progress:",  total=npoints, disable=None):
            for j, b in enumerate(betas):
                model_params = {k: trained_weights[k] + a*dir1[k] + b*dir2[k] for k in trained_weights}
                z[i, j] = _loss_at(model, model_params, data[:5000], targets[:5000], lossF)

    return alphas, betas, z