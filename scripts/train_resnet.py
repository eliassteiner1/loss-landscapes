import torch
import torch.nn as nn
import tyro

from loss_landscapes.datagen.io import load_data
from loss_landscapes.networks.io import save_model
from loss_landscapes.paths import PROJ_ROOT
from loss_landscapes.networks.resnet import SmallResNet, SmallResNetConfig
from loss_landscapes.train.loop import training_loop
from loss_landscapes.train.setup import DataSplit
from loss_landscapes.train.setup import TrainingConfig
from loss_landscapes.train.setup import get_device


# as a main or put this into the executed block??
# this is what will run piece by piece in the simple demo notebook
def main(cfg: TrainingConfig, mcfg: SmallResNetConfig, dataset_name: str = "ds_fashion.h5"):
     
    device = get_device()
    data, targets = load_data(dataset_name)
    
    dsp = DataSplit(
        n_total=len(data),
        n_train = int(len(data) * cfg.train_split)
    )
    
    # normalize the data; with statistics determined from the training part only, but applied to entire dataset. This way, no information leaks from the training part to the validation part.
    data_mean = data[:dsp.n_train].mean() # scalar
    data_sdev = data[:dsp.n_train].std() # scalar
    data = (data - data_mean) / data_sdev
    
    # setup main objects before training
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    network = SmallResNet(mcfg).to(device)
    optimizer = torch.optim.Adam(params=network.parameters(), lr=1e-3)
    data = torch.as_tensor(data, dtype=torch.float32)[:, None, :, :].to(device) # add channel singleton dim for conv
    targets = torch.as_tensor(targets, dtype=torch.long).to(device) # nn.CrossEntropyLoss needs int

    # run training loop (includes validation)
    best_state_dict, _ = training_loop(
        cfg,
        dsp,
        data, targets,
        device,
        network,
        optimizer,
        loss_function
    )

    # store the model
    save_model(
        f"resnet_b{mcfg.n_blocks_per_stage:02d}_{"skip" if mcfg.use_skip else "noskip"}.pt",
        best_state_dict, 
        mcfg, 
        data_mean=data_mean,
        data_sdev=data_sdev,
    )

   
if __name__ == "__main__":
    tyro.cli(main)
    
    

