
import torch
import torch.nn

from loss_landscapes.paths import PROJ_ROOT
from loss_landscapes.networks.resnet import SmallResNet, SmallResNetConf


def load_model(name: str):
    ckpt = torch.load(PROJ_ROOT / "checkpoints/resnet_b9_noskip.pt")
    model = SmallResNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    


def main():
    ...
  


if __name__ == "__main__":
    
    main()