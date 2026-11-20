from loss_landscapes.networks.resnet import SmallResNet, SmallResNetConf
from loss_landscapes.paths import PROJ_ROOT


import torch


from dataclasses import asdict


def save_model(name: str, state_dict, CfgModel, data_mean: float, data_sdev: float):
    path = PROJ_ROOT / "checkpoints" / name
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()}, # always save "on cpu"
            "config": asdict(CfgModel),
            "data_mean": float(data_mean), # saving with torch only acceppts pure types or tensors!
            "data_sdev": float(data_sdev), # saving with torch only acceppts pure types or tensors!
        },
        path,
    )
    print(f"> saved model snapshot {path}")


def load_model(name: str):
    path = PROJ_ROOT / "checkpoints" / name
    
    snapshot = torch.load(path, map_location="cpu") # always load "on cpu"
    CfgModel = SmallResNetConf(**snapshot["config"])
    network = SmallResNet(CfgModel)
    network.load_state_dict(snapshot["state_dict"])
    print(f"> loaded model snapshot {path}")

    return network, snapshot["data_mean"], snapshot["data_sdev"]
    
