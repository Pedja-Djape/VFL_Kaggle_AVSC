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

# class Net(nn.Module):

#     def __init__(self,n_inputs,output_dim) -> None:
#         super(Net, self).__init__()
#         self.middle_layer = (n_inputs + output_dim) // 2
#         self.output_dim = output_dim

#         self.fc1 = nn.Linear(n_inputs, self.middle_layer)
#         self.fc3 = nn.Linear(self.middle_layer,output_dim)

#     def forward(self,x):
#         h = F.relu(self.fc1(x))
#         h = self.fc3(h)
#         return sigmoid(h)


