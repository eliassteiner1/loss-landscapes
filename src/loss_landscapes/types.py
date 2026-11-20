from collections.abc import Callable
from typing import TypedDict
import torch

type Params = dict[str, torch.Tensor] # trainable parameters only: one point in parameter space
type StateDict = dict[str, torch.Tensor] # params + buffers: for saving/loading

class Checkpoint(TypedDict):
    state_dict: StateDict
    config: dict
    data_mean: float
    data_sdev: float

type LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]