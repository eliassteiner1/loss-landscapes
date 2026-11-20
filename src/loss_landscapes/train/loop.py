from tqdm.auto import tqdm

import copy

from loss_landscapes.train.setup import DataSplit
from loss_landscapes.train.setup import TrainingConfig
from loss_landscapes.types import LossFunction, StateDict

import torch
import torch.nn as nn



def _get_batch_indices(indices: torch.Tensor, start: int, size: int) -> torch.Tensor:
    return indices[start:start+size]


def _do_validation(
    dsp: DataSplit,
    data: torch.Tensor, targets: torch.Tensor,
    device: str,
    network: nn.Module,
    loss_function: LossFunction,
) -> tuple[float, float]:

    valid_indices = torch.arange(dsp.n_valid, device=device) + dsp.n_train
    network.eval()
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for batch_start in range(0, dsp.n_valid, 1_000):
            batch_indices = _get_batch_indices(valid_indices, batch_start, 1_000)

            pred = network(data[batch_indices])
            loss = loss_function(pred, targets[batch_indices])
            total_loss += loss.item() * len(batch_indices)

            pred_labels = torch.argmax(pred, dim=1)
            total_correct += (pred_labels == targets[batch_indices]).sum().item()

    return total_loss/dsp.n_valid, total_correct/dsp.n_valid


def training_loop(
    cfg: TrainingConfig, dsp: DataSplit,
    data: torch.Tensor, targets: torch.Tensor,
    device: str, 
    network: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    loss_function: LossFunction,
) -> tuple[StateDict | None, dict[str, list[float]]]:

    history = {"train_loss": [], "valid_loss": [], "valid_accy": []}
    best_loss = float("inf")
    best_state_dict = None
    
    if hasattr(network, "mcfg"):
        print(f"> training model with use_skip={network.mcfg.use_skip}")
    else: 
        print(f"> training model ....")
        
    for E in tqdm(range(cfg.n_epochs), desc="  progress: ", position=0, disable=None): # disable in notebook

        # ===== training =====
        train_indices = torch.randperm(dsp.n_train, device=device)
        network.train()
        total_loss = 0.0

        for batch_start in range(0, dsp.n_train, cfg.batch_size):
            batch_indices = _get_batch_indices(train_indices, batch_start, cfg.batch_size)

            optimizer.zero_grad()
            pred = network(data[batch_indices])
            loss = loss_function(pred, targets[batch_indices])
            loss.backward()
            optimizer.step()

            total_loss += loss.detach() * len(batch_indices)

        loss_mean_train = (total_loss/dsp.n_train).item() # one gpu sync per epoch

        # ===== validation =====
        loss_mean_valid, accuracy_mean_valid = _do_validation(
            dsp,
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
            best_state_dict = copy.deepcopy(network.state_dict()) # deepcopy! otherwise the copy will continue to change

    return best_state_dict, history