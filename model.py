import torch.nn as nn
from torch import sigmoid
import torch
import torch.nn.functional as F


# each client in a vertical setting will have a different
# architecture


class Net(nn.Module):

    def __init__(self,sizes) -> None:
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_features, out_features) for in_features,out_features in zip(sizes,sizes[1:])
        ])
    
    def forward(self,x):
        for i,layer in enumerate(self.layers):
            # sigmoid only if output layers
            fn = F.relu if i < (len(self.layers)-1) else sigmoid
            x = fn(layer(x))
        return x



