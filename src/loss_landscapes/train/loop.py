from tqdm.auto import tqdm

import copy

from loss_landscapes.train.setup import DataSplit
from loss_landscapes.train.setup import TrainingConfig


import torch
import torch.nn as nn


from collections.abc import Callable


def _get_batch_indices(indices, start, size):
    return indices[start:start+size]


def _do_validation(
    DsSplit: DataSplit,

    data: torch.Tensor, targets: torch.Tensor,
    device: str,
    network: nn.Module,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):

    valid_indices = torch.arange(DsSplit.n_valid, device=device) + DsSplit.n_train
    network.eval()
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for batch_start in range(0, DsSplit.n_valid, 1_000):
            batch_indices = _get_batch_indices(valid_indices, batch_start, 1_000)

            pred = network(data[batch_indices])
            loss = loss_function(pred, targets[batch_indices])
            total_loss += loss.item() * len(batch_indices)

            pred_labels = torch.argmax(pred, dim=1)
            total_correct += (pred_labels == targets[batch_indices]).sum()

    return total_loss/DsSplit.n_valid, total_correct/DsSplit.n_valid


def training_loop(
    CfgTrain: TrainingConfig,
    DsSplit: DataSplit,
    data: torch.Tensor, targets: torch.Tensor,
    device: str,
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):

    history = {"train_loss": [], "valid_loss": [], "valid_accy": []}
    best_loss = float("inf")
    best_snapshot = None
    
    print(f"> training model with {network._cfg_backup.use_skip=}")
    for E in tqdm(range(CfgTrain.n_epochs), desc="  progress: ", position=0, disable=None): # disable in notebook

        # ===== training =====
        train_indices = torch.randperm(DsSplit.n_train, device=device)
        network.train()
        total_loss = 0.0

        for batch_start in range(0, DsSplit.n_train, CfgTrain.batch_size):
            batch_indices = _get_batch_indices(train_indices, batch_start, CfgTrain.batch_size)

            optimizer.zero_grad()
            pred = network(data[batch_indices])
            loss = loss_function(pred, targets[batch_indices])
            loss.backward()
            optimizer.step()

            total_loss += loss.detach() * len(batch_indices)

        loss_mean_train = (total_loss/DsSplit.n_train).item() # one gpu sync per epoch

        # ===== validation =====
        loss_mean_valid, accuracy_mean_valid = _do_validation(
            DsSplit,
            data, targets,
            device,
            network,
            loss_function
        )

        # ===== loggin, saving model snapshot, etc =====
        history["train_loss"].append(loss_mean_train)
        history["valid_loss"].append(loss_mean_valid)
        history["valid_accy"].append(accuracy_mean_valid)

        tqdm.write(
            f"  Epoch {E+1:02d}: "
            f"loss_train = {loss_mean_train:.4f} | "
            f"loss_valid = {loss_mean_valid:.4f} | "
            f"accuracy_valid = {accuracy_mean_valid:.2f}"
        )

        if loss_mean_valid < best_loss:
            best_loss = loss_mean_valid
            best_snapshot = copy.deepcopy(network.state_dict()) # deepcopy! otherwise the copy will continue to change

    return best_snapshot, history