

import client
import utils
import model
import strategy as stgy
from torch.optim import Adam
import flwr as fl

def client_fn(cid):
    """
    Create a flower clinet representing a single organization.
    """
    dl = DATA['data'][int(cid)]
    
    net = model.Net(dl.dataset.X.shape[-1], 4)
    return client.FlowerClient(cid,net, dl, optimizer=Adam(net.parameters(),lr=1e-6))

if __name__ == "__main__":
    BATCH_SIZE = 64
    NUM_CLIENTS = 3
    NUM_ROUNDS = 1000
    DATA = utils.load_datasets("../data/train_data.csv", batch_size=BATCH_SIZE)

    strategy = stgy.SplitVFL(
        num_clients=NUM_CLIENTS, 
        batch_size=BATCH_SIZE, 
        dim_input= 12, # 4 outputs for three clients
        labels = DATA['labels']
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(
            num_rounds = NUM_ROUNDS
        ),
        strategy=strategy,
    )