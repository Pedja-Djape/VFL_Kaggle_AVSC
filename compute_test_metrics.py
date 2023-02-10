from model import Net
import torch
from utils import ShoppersDataset,ClientIdentifier
from sklearn import metrics
import matplotlib.pyplot as plt
import argparse
from pickle import load

# load a given model from it's picked form
def load_model(dim_input, dim_output, cid=None):
    model = Net(dim_input,dim_output)
    if cid is None:
        model.load_state_dict(torch.load("./models/global_model.pt"))
    else:
        model.load_state_dict(torch.load(f"./models/model_{cid}.pt"))
    model.eval()
    return model

# get a given client's test data
def load_client_data(data_dict, cid, ci):
    client_type = ci.get_client_from_cid(cid=cid)
    return data_dict['data'][client_type]['test']

# get input dimension of a client model
def get_client_input_dim(data_dict, cid, ci):
    client_type = ci.get_client_from_cid(cid=cid)
    return data_dict['data'][client_type]['test'].dataset.X.shape[-1]

# get all client datasets, models, and targets
def get_client_info(num_clients, infile, ci):

    # load data saved earlier (in Server.py)
    with open(infile,'rb') as f:
        data = load(f)
    # get labels
    labels = data['test_labels']

    # get client_data, clients loaded in order of increasing cid 
    client_data = [
        load_client_data(data, i, ci=ci) for i in range(num_clients)
    ]

    client_models = [
        load_model(
            dim_input = get_client_input_dim(data, i, ci=ci), 
            dim_output = 6,
            cid = i
        ) for i in range(num_clients)
    ]
    return client_data, client_models, labels

def get_client_order(cids):
    rval = {}
    for cid in cids:
        with open(f"./models/model_{cid}_info.pt","rb") as f:
            rval[cid] = load(f)['order']
    return rval


def main(num_clients, infile):

    ci = ClientIdentifier()
    
    client_data, client_models, labels_dl = get_client_info(num_clients, infile, ci=ci)

    labels = iter(labels_dl)

    client_iters = [iter(dl) for dl in client_data]

    global_model = load_model(dim_input = 6*num_clients, dim_output = 1)

    criterion = torch.nn.BCELoss()
    
    pred_scores = []
    
    client_order = get_client_order(cids=[0,1,2])

    with torch.no_grad():
        while True:
            try:
                # get client inputs
                Client0Inputs = next(client_iters[0]).float()
                Client1Inputs = next(client_iters[1]).float()
                Client2Inputs = next(client_iters[2]).float()
                # get labels
                ls = next(labels)
                # get embeddings
                Client0Embeddings = client_models[0](Client0Inputs).detach()
                Client1Embeddings = client_models[1](Client1Inputs).detach()
                Client2Embeddings = client_models[2](Client2Inputs).detach()

                input_tensor = [None for i in range(3)]
                input_tensor[client_order[0]] = Client0Embeddings
                input_tensor[client_order[1]] = Client1Embeddings
                input_tensor[client_order[2]] = Client2Embeddings
                # create input to global model
                gbl_mdl_inputs = torch.cat(input_tensor, dim = 1 )
                # get outputs
                gbl_mdl_outputs = global_model(gbl_mdl_inputs)
                
                pred_scores += gbl_mdl_outputs.squeeze().tolist()
            except StopIteration:
                # print("Successfully iterated through test set", len(pred_scores) == labels_dl.dataset.X.shape[0])
                break
    # Get AUC of ROC
    fpr, tpr, thresholds = metrics.roc_curve(y_true=labels_dl.dataset.X.squeeze(), y_score = pred_scores)
    roc_auc = metrics.auc(fpr,tpr)
    display = metrics.RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc)
    display.plot()
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numclients",type=int)
    parser.add_argument("-d", "--datafile",type=str)

    args = parser.parse_args()

    NUM_CLIENTS = args.numclients
    infile = args.datafile
    
    main(
        num_clients=NUM_CLIENTS,
        infile=infile
    )
    pass



