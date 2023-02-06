import pandas as pd 
from utils import ShoppersDataset,ClientIdentifier
from torch.utils.data import DataLoader
from model import Net
import torch
from pickle import load
import argparse

# load a given model from it's picked form
def load_model(dim_input, dim_output, cid=None):
    model = Net(dim_input,dim_output)
    if cid is None:
        model.load_state_dict(torch.load("./models/global_model.pt"))
    else:
        model.load_state_dict(torch.load(f"./models/model_{cid}.pt"))
    model.eval()
    return model

def get_client_order(cids):
    rval = {}
    for cid in cids:
        with open(f"./models/model_{cid}_info.pt","rb") as f:
            rval[cid] = load(f)['order']
    return rval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-gblr","--globallr",type=float)
    parser.add_argument("-bs","--batchsize",type=int)
    parser.add_argument("-clrs","--clientlrs",nargs='+',type=float)
    parser.add_argument("-nr","--numrounds",type=int)
    parser.add_argument("-s","--scheduler")

    args       = parser.parse_args()
    gblr       = args.globallr 
    bs         = args.batchsize 
    clrs       = args.clientlrs
    num_rounds = args.numrounds
    scheduler  = args.scheduler
    

    test_data = pd.read_csv("../change_data/tmp_test_data.csv")

    comp_cols, brand_cols, cat_cols = [],[],[]
    for i,col in enumerate(test_data.columns):
        cnt = -1
        if 'brand' in col: 
            cnt += 1
            brand_cols.append(col)
        if 'cat' in col:
            cnt += 1
            cat_cols.append(col)
        if 'comp' in col:
            cnt += 1
            comp_cols.append(col)
        if cnt == -1:
            comp_cols.append(col)
    
    company_client = test_data[comp_cols].to_numpy()
    brand_client = test_data[brand_cols].to_numpy()
    category_client = test_data[cat_cols].to_numpy()

    company_dl = DataLoader(
        dataset = ShoppersDataset(company_client),
        batch_size=company_client.shape[0],
        shuffle=False
    )
    brand_dl = DataLoader(
        dataset = ShoppersDataset(brand_client),
        batch_size=brand_client.shape[0],
        shuffle=False
    )
    category_dl = DataLoader(
        dataset = ShoppersDataset(category_client),
        batch_size=category_client.shape[0],
        shuffle=False
    )

    ci = ClientIdentifier()
    company_cid = ci.get_cid_from_client(client_type='company')
    brand_cid = ci.get_cid_from_client(client_type='brand')
    category_cid = ci.get_cid_from_client(client_type='category')

    client_outputs = 6

    company_model = load_model(dim_input=company_client.shape[-1], dim_output=client_outputs,cid = company_cid)
    brand_model = load_model(dim_input=brand_client.shape[-1], dim_output=client_outputs, cid = brand_cid)
    category_model = load_model(dim_input=category_client.shape[-1], dim_output=client_outputs, cid = category_cid)

    global_model = load_model(dim_input = client_outputs*3, dim_output = 1)

    client_order = get_client_order([company_cid,brand_cid,category_cid])

    with torch.no_grad():
        company_outputs = company_model(next(iter(company_dl)).float())
        brand_outputs = brand_model(next(iter(brand_dl)).float())
        category_outputs = category_model(next(iter(category_dl)).float())

        gbl_model_inputs = [None for i in range(3)]

        gbl_model_inputs[client_order[company_cid]] = company_outputs
        gbl_model_inputs[client_order[brand_cid]] = brand_outputs
        gbl_model_inputs[client_order[category_cid]] = category_outputs

        gbl_model_inputs = torch.cat(gbl_model_inputs,dim=1)

        gbl_model_outputs = global_model(gbl_model_inputs)

        preds = gbl_model_outputs.numpy()
    
    # do the client outputs have sigmoid applied
    client_sigmoid_output = 1
    ids = pd.read_csv("../data/testHistory.csv").id 
    df = pd.DataFrame(data={"id": ids, "repeatProbability": preds.squeeze()})

    clrs_string = ''.join(str(lr) for lr in clrs)
    df.to_csv(f'./eval/evaltest_co_{client_outputs}_gblr_{gblr}_bs_{bs}_clrs_{clrs_string}_nr_{num_rounds}_csmo_{client_sigmoid_output}_sched_{scheduler}_nd_1.csv',index=False)



    
    