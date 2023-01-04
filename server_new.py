import strategy_new as stgy
import pickle
from utils import ShoppersDataset,save_data
import time

from torch.optim import Adam
from torch.utils.data import DataLoader
import flwr as fl


if __name__ == "__main__":
    BATCH_SIZE = 32
    NUM_CLIENTS = 3
    NUM_ROUNDS = 5000
    outfile = './data.pt'
    infile = './data.pt'

    
    DATA = save_data(data_path='../data/train_data.csv', batch_size=BATCH_SIZE, outfile=outfile)

    if 'batch_size' not in DATA or DATA['batch_size'] != BATCH_SIZE:
        print(f"Saving data with new batch_size: {BATCH_SIZE}")
        DATA = save_data(data_path='../data/train_data.csv', batch_size=BATCH_SIZE, outfile=outfile)
    
    
    strategy = stgy.SplitVFL(
        num_clients=NUM_CLIENTS, 
        batch_size=BATCH_SIZE, 
        dim_input= 18, # 6 outputs for three clients
        labels = DATA['labels']
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),
        strategy = strategy
    )
