import flwr as fl 
import torch
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

from flwr.common import parameters_to_ndarrays
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pickle as pi




class FlowerClient(fl.client.NumPyClient):
    def __init__(self,cid, net, trainloader,testloader, optimizer):
        super(FlowerClient,self).__init__()
        # client id
        self.cid = cid
        # client net
        self.net = net
        # train iterator
        self.trainloader = trainloader
        self.trainiter = iter(trainloader)

        self.test_loader = testloader 
        self.testiter = iter(self.test_loader)

        # optimizer (Adamm, etc.)
        self.optimizer = optimizer
        # outputs from previous round
        self.outputs = None
        

    def get_parameters(self, config):
        return [
            val.cpu.numpy() for _,val in self.net.state_dict().items()
        ]

    def set_parameters(self, parameters):
        return

    def fit(self,parameters,config):
        # Read values from config
        server_round = config['server_round']
        # get batch
        try:
            X = next(self.trainiter)
        except StopIteration:
            self.trainiter = iter(self.trainloader)
            X = next(self.trainiter)
        outputs = self.net(X.float())
        self.outputs = outputs
        return [x for x in outputs.detach().numpy()], 1, {}

    def evaluate(self, parameters, config):
        self.outputs.backward(torch.tensor(np.array(parameters)))
        self.optimizer.step()

        try:
            X = next(self.testiter)
        except StopIteration:
            self.testiter = iter(self.test_loader)
            X = next(self.testiter)
        
        with torch.no_grad():
            outputs = self.net(X.float()).numpy()

        return 0., 0, {'test_embeddings': pi.dumps([ x for x in outputs])}
    
if __name__ == "__main__":
    import argparse
    import pickle
    from utils import ShoppersDataset
    import model
    from torch.optim import Adam

    parser = argparse.ArgumentParser()
    parser.add_argument("cid")
    args = parser.parse_args()
    
    

    with open('data.pt','rb') as f:
        data = pickle.load(f)
    
    train_dataloader = data['data']['train'][int(args.cid)]
    test_dataloader = data['data']['test'][int(args.cid)]
    
    torch.manual_seed(0)
    model = model.Net(train_dataloader.dataset.X.shape[-1], 6)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(
            cid=str(args.cid), 
            net = model, 
            trainloader=train_dataloader, 
            testloader=test_dataloader, 
            optimizer = Adam(model.parameters(), lr=1e-8)
        )
    )
    

