from loss_landscapes.paths import PROJ_ROOT


import torch


from dataclasses import asdict


def save_model(snapshot, CfgModel, name: str):
    torch.save(
        {
            "model": snapshot,
            "config": asdict(CfgModel)
        },
        PROJ_ROOT / "checkpoints" / name,
    )
    print(f"> saved model snapshot {PROJ_ROOT / "checkpoints" / name}")
    
# TODO: load model