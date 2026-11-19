import torch
import torch.nn as nn
import tyro

from loss_landscapes.datagen.io import load_data
from loss_landscapes.networks.io import save_model
from loss_landscapes.paths import PROJ_ROOT
from loss_landscapes.networks.resnet import SmallResNet, SmallResNetConf
from loss_landscapes.train.loop import training_loop
from loss_landscapes.train.setup import DataSplit
from loss_landscapes.train.setup import TrainingConfig
from loss_landscapes.train.setup import get_device


# as a main or put this into the executed block??
# this is what will run piece by piece in the simple demo notebook
def main(CfgTrain: TrainingConfig, CfgModel: SmallResNetConf, dataset_path: str = "data/ds_fashion.h5"):
     
    device = get_device()
    X, y = load_data(PROJ_ROOT / dataset_path)
    
    DsSplit = DataSplit(
        n_total=len(X),
        n_train = int(len(X) * CfgTrain.train_split)
    )
    
    # normalize the data; with statistics determined from the training part only, but applied to entire dataset. This way, no information leaks from the training part to the validation part.
    mean = X[:DsSplit.n_train].mean() # scalar
    sdev = X[:DsSplit.n_train].std() # scalar
    X = (X - mean) / sdev
    
    # setup main objects before training
    lossF = nn.CrossEntropyLoss(reduction="mean")
    model = SmallResNet(CfgModel).to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    X = torch.as_tensor(X, dtype=torch.float32)[:, None, :, :].to(device) # add 1 channel singleton dim for conv layers
    y = torch.as_tensor(y, dtype=torch.long).to(device) # nn.CrossEntropyLoss needs int

    # run training loop (includes validation)
    best_snapshot, _ = training_loop(
        CfgTrain,
        DsSplit,
        X, y,
        device,
        model,
        optim,
        lossF
    )

    # store the model
    save_model(
        best_snapshot, CfgModel, 
        f"resnet_b{CfgModel.n_blocks_per_stage:02d}_{"skip" if CfgModel.use_skip else "noskip"}.pt",
    )

   
if __name__ == "__main__":
    tyro.cli(main)
    
    

