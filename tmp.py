import torch.nn as nn
from torch import sigmoid
import torch

from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F

import numpy as np

# each client in a vertical setting will have a different
# architecture


class Dummy(nn.Module):
    """
        Local client model:
            Input Layer --> Middle Layer --> Output Layer (activated)
    """
    def __init__(self, n_inputs, output_dim,is_global=False) -> None:
        super(Dummy, self).__init__()
        self.fc1 = nn.Linear(n_inputs, output_dim)
        self.is_global = is_global
    
    def forward(self,x):
        h = F.relu(self.fc1(x))
        if self.is_global:
            h = torch.sigmoid(h)
        return h

class CD(Dataset):
    def __init__(self,X):
        self.X = X
        # self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index,:]#,self.y[index]


def get_data(csv_path):
    df = pd.read_csv(csv_path)
    y = df['repeater'].values.reshape((-1,1))
    X = df.drop("repeater",axis=1).to_numpy()
    return X,y

np.random.seed(0)
c1 = Dummy(10,2)
c2 = Dummy(5,2)

g = Dummy(4, 1,is_global=True)

tmp = [1,2,3,4,5,6,7,8,9,10]

d1 = np.array(
    [tmp,
    [i*2 for i in tmp],
    [i*4 for i in tmp]]
)


d2 = np.array(
    [[1 for i in tmp[:5]],
    [2 for i in tmp[:5]],
    [3 for i in tmp[:5]]]
)


y2 = torch.tensor(np.array([1,0,0]),dtype=torch.float32)

ds1 = DataLoader(CD(d1),batch_size=30)
ds2 = DataLoader(CD(d2),batch_size=30)

opt1 = torch.optim.Adam(c1.parameters(),lr=0.01)
opt2 = torch.optim.Adam(c2.parameters(),lr=0.01)
optg = torch.optim.Adam(g.parameters(),lr=0.01)

criterion = torch.nn.BCELoss()

opt1.zero_grad(); opt2.zero_grad()

o1 = c1(next(iter(ds1)).float())
o2 = c2(next(iter(ds2)).float())

oo1 = o1.detach().numpy()
oo2 = o2.detach().numpy()

client_tensors = [
    torch.tensor(oo1,dtype=torch.float32,requires_grad=True),
    torch.tensor(oo2,dtype=torch.float32,requires_grad=True)
]

# running client_tensors[i].grad as this point would return None

"""
Recall that when initializing optimizer you explicitly tell it what parameters (tensors) of the model it should be updating. 
The gradients are "stored" by the tensors themselves (they have a grad and a requires_grad attributes) once you call backward() 
on the loss. After computing the gradients for all tensors in the model, calling optimizer.step() makes the optimizer iterate 
over all parameters (tensors) it is supposed to update and use their internally stored grad to update their values.
"""

input_tensor = torch.cat(client_tensors,1)

global_output = g(input_tensor)

optg.zero_grad()

loss = criterion(global_output.squeeze(1),
                y2
            )

loss.backward()
optg.step()

for k,v in c1.named_parameters():
    print(k,v.size())
# # running client_tensors[i].grad at this point would return [3,2]
o1.backward(client_tensors[0].grad)
o2.backward(client_tensors[1].grad)

opt1.step()
print(c1.parameters())
opt2.step()

