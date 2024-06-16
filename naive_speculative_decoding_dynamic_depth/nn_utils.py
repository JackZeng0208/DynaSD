import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List



class ResnetBlock(nn.Module):
    def __init__(self, fc: nn.Module):
        self.fc = fc
        self.act = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        return x + self.act(self.fc(x))
    

class DecicisionModel(nn.Module):
    def __init__(self, in_dim: int, net_arch: List[int], out_dim: int, feature_dim: int = 0):
        self.act = nn.ReLU()
        layers = [nn.Linear(in_dim + feature_dim, net_arch[0])]
        for i in range(1, len(net_arch)):
            layers.append(ResnetBlock(nn.Linear(net_arch[i - 1], net_arch[i])))
        layers.append(nn.Linear(net_arch[-1]))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, hidden_feature: Optional[torch.Tensor]):
        if hidden_feature:
            x = torch.concat((x, hidden_feature), dim=-1) 
        return self.net(x)



