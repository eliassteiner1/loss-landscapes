from loss_landscapes.networks.resnet import SmallResNet, SmallResNetConfig
from loss_landscapes.types import StateDict, Checkpoint
from loss_landscapes.paths import PROJ_ROOT


import torch


from dataclasses import asdict


def save_model(
    name: str, 
    state_dict: StateDict, 
    mcfg: SmallResNetConfig, 
    data_mean: float, 
    data_sdev: float
) -> None:
    
    path = PROJ_ROOT / "checkpoints" / name
    torch.save(
        Checkpoint(
            state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}, # always save "on cpu"
            config = asdict(mcfg),
            data_mean = float(data_mean), # saving with torch only accepts pure types or tensors!
            data_sdev = float(data_sdev), # saving with torch only accepts pure types or tensors!
        ),        
        path,
    )
    print(f"> saved model snapshot {path}")


def load_model(name: str) -> tuple[SmallResNet, float, float]:
    path = PROJ_ROOT / "checkpoints" / name
    
    checkpoint: Checkpoint = torch.load(path, map_location="cpu") # always load "on cpu"
    mcfg = SmallResNetConfig(**checkpoint["config"])
    network = SmallResNet(mcfg)
    network.load_state_dict(checkpoint["state_dict"])
    print(f"> loaded model snapshot {path}")

    return network, checkpoint["data_mean"], checkpoint["data_sdev"]
    
