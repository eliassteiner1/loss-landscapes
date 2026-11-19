import torch
import torch.nn as nn
from dataclasses import dataclass


class Block(nn.Module):
    def __init__(self, ch: int, skip: bool):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=ch)
        self.conv2 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=ch)
        self.skip = skip
        self.activ = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activ(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.skip:
            return self.activ(out + x)
        else:
            return self.activ(out)

@dataclass
class SmallResNetConf:
    n_in_channels: int = 1
    n_hidden_channels: int = 16
    n_blocks_per_stage: int = 9
    n_stages: int = 3 # each stage halves resolution!
    use_skip: bool = True 
    n_classes: int = 10
    
class SmallResNet(nn.Module):
    def __init__(self, Cfg: SmallResNetConf):
        super().__init__()
        self._cfg_backup = Cfg
        
        # this is the main chain of blocks > multiple stages, each a bunch of blocks with the same params
        layers = []
        for s in range(Cfg.n_stages):
            
            # doubles the number of hodden channels for each progressive stage (as the resolution is halved)
            c = Cfg.n_hidden_channels * 2**s
            
            # Transition-Layer: this goes from input img channels to n_hidden_channels that are used in blocks
            if s == 0:
                layers += [
                    nn.Conv2d(Cfg.n_in_channels, c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_features = c),
                    nn.ReLU(), 
                ]
            
            # Transition-Layer: go from one stage to the next -> Res/2 while ch*2, so a transition layer is needed
            if s > 0:
                layers += [
                    nn.Conv2d(c//2, c, 3, stride=2, padding=1, bias=False), 
                    nn.BatchNorm2d(c),
                    nn.ReLU()
                ]
                
            # stage blocks, all the same (create a new block! not just copying reference with []*b_p_s!)
            layers += [Block(c, Cfg.use_skip) for _ in range(Cfg.n_blocks_per_stage)]  
        
        self.body = nn.Sequential(*layers)
        
        # the output with mlp to get to a flat n_classes output
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(Cfg.n_hidden_channels * 2**(Cfg.n_stages-1), Cfg.n_classes)
        )

    def forward(self, x):
        return self.head(self.body(x))
        
 