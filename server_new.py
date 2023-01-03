import strategy_new as stgy
import pickle
from utils import ShoppersDataset

from torch.optim import Adam
import flwr as fl


if __name__ == "__main__":
    BATCH_SIZE = 64
    NUM_CLIENTS = 3
    NUM_ROUNDS = 60

    with open('data.pt','rb') as f:
        DATA = pickle.load(f)


    strategy = stgy.SplitVFL(
        num_clients=NUM_CLIENTS, 
        batch_size=BATCH_SIZE, 
        dim_input= 12, # 4 outputs for three clients
        labels = DATA['labels']
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),
        strategy = strategy
    )
