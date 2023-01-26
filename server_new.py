import strategy_new as stgy
import pickle
from utils import ShoppersDataset,load_datasets,save_data
import time
from torch import save

from torch.optim import Adam
from torch.utils.data import DataLoader
import flwr as fl

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-bs","--batchsize",type=int)
    parser.add_argument("-n", "--numclients",type=int)
    parser.add_argument("-do","--dataoutput",type=str)
    parser.add_argument("-nr","--numrounds",type=int)
    parser.add_argument("-f","--trainfile",type=str)

    args = parser.parse_args()

    

    # batch size and number of clients
    BATCH_SIZE = args.batchsize
    NUM_CLIENTS = args.numclients
    # number of batches to iterate through
    NUM_ROUNDS = args.numrounds
    outfile = args.dataoutput
    infile = args.trainfile
    
    # get data and save
    DATA = save_data(data_path=infile, batch_size=BATCH_SIZE,outfile=outfile)

    # create strategy
    Strategy = stgy.SplitVFL(
        num_clients=NUM_CLIENTS, 
        batch_size=BATCH_SIZE, 
        dim_input= 18, # 6 outputs for three clients
        train_labels = DATA['train_labels'],
        test_labels=DATA['test_labels'],
    )
    # start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),
        strategy = Strategy
    )

    # get and save global model for testing purposes.
    global_model = Strategy.get_model()
    save(global_model, "./models/global_model.pt")
    
    import matplotlib.pyplot as plt 

    fig, axs = plt.subplots(1,2,sharey=True)
    
    axs[0].plot(Strategy.train_acc,label='Train Accuracy')
    axs[1].plot(Strategy.test_acc, label='Test Accuracy')
    axs[1].plot(Strategy.test_f1,label='Test F1-Score')
    axs[0].legend()
    axs[1].legend()
    plt.show()
