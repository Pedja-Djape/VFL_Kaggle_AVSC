import strategy_new as stgy
import pickle
from utils import ShoppersDataset,load_datasets,save_data
import time

from torch.optim import Adam
from torch.utils.data import DataLoader
import flwr as fl


if __name__ == "__main__":
    BATCH_SIZE = 32
    NUM_CLIENTS = 3
    NUM_ROUNDS = 30008
    outfile = './data.pt'
    infile = './data.pt'

    
    DATA = save_data(data_path='../data/train_data.csv', batch_size=BATCH_SIZE,outfile=outfile)
    strategy = stgy.SplitVFL(
        num_clients=NUM_CLIENTS, 
        batch_size=BATCH_SIZE, 
        dim_input= 18, # 6 outputs for three clients
        train_labels = DATA['train_labels'],
        test_labels=DATA['test_labels']
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),
        strategy = strategy
    )
    
    import matplotlib.pyplot as plt 

    fig, axs = plt.subplots(1,2,sharey=True)
    
    axs[0].plot(strategy.train_acc,label='Train Accuracy')
    axs[1].plot(strategy.test_acc, label='Test Accuracy')
    axs[1].plot(strategy.test_f1,label='Test F1-Score')
    axs[0].legend()
    axs[1].legend()
    plt.show()
