import flwr as fl 
import torch
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

from flwr.common import parameters_to_ndarrays
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

OUTPUTS = None

import pickle
torch.manual_seed(0)
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,cid, net, trainloader,optimizer):
        super(FlowerClient,self).__init__()
        # client id
        self.cid = cid
        # client net
        self.net = net
        # train iterator
        self.trainiter = iter(trainloader)
        # optimizer (Adamm, etc.)
        self.optimizer = optimizer
        # outputs from previous round
        self.outputs = []
        

    def get_parameters(self, config):
        return [
            val.cpu.numpy() for _,val in self.net.state_dict().items()
        ]

    def set_parameters(self, parameters):
        return

    def fit(self,parameters,config):
        # Read values from config
        server_round = config['server_round']
        print(f"(Client {self.cid}), round {server_round} fit, config: {config}")
        # get batch
        X = next(self.trainiter)
        outputs = self.net(X.float())
        torch.save(outputs,f"outputs_{self.cid}.pt")
        # self.outputs = outputs
        # self.outputs.append(4)
        # global OUTPUTS
        # OUTPUTS = outputs 
        # if OUTPUTS == None:
        #     print("\n\n\nNONE\n\n\n")
        # returning outputs of model to server (list of ndarrays)
        return [x for x in outputs.detach().numpy()], 1, {}

    def evaluate(self, parameters, config):
    
        # print(f"\n\nclient.evaluate ... SERVER_ROUND: {config['server_round']} {outputs}\n\n")
        # self.outputs.backward(
        #     torch.tensor(parameters_to_ndarrays(parameters[0]))
        # )
        # global OUTPUTS
        # if OUTPUTS == None:
        #     print("\n\n\nNONE\n\n\n")
        # OUTPUTS.backward(
        #     torch.tensor(np.array(parameters)),
        #     retain_graph=True
        # )
        o = torch.load(f'outputs_{self.cid}.pt')
        o.backward(torch.tensor(np.array(parameters)))
        self.optimizer.step()
        return .0, 0, {}
    
