import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, dim_in: int = 2, dim_hidden: int = 64, dim_out: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_out)
        self.activ = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        x = self.fc3(x)
        return x