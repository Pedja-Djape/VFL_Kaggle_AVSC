import pandas as pd 
from utils import ShoppersDataset,ClientIdentifier
from torch.utils.data import DataLoader
from model import Net
import torch

# load a given model from it's picked form
def load_model(dim_input, dim_output, cid=None):
    model = Net(dim_input,dim_output)
    if cid is None:
        model.load_state_dict(torch.load("./models/global_model.pt"))
    else:
        model.load_state_dict(torch.load(f"./models/model_{cid}.pt"))
    model.eval()
    return model



if __name__ == "__main__":

    test_data = pd.read_csv("../data/tmp_test_data.csv")

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

    company_model = load_model(dim_input=company_client.shape[-1], dim_output=6,cid = company_cid)
    brand_model = load_model(dim_input=brand_client.shape[-1], dim_output=6, cid = brand_cid)
    category_model = load_model(dim_input=category_client.shape[-1], dim_output=6, cid = category_cid)

    global_model = load_model(dim_input = 6*3, dim_output = 1)

    with torch.no_grad():
        company_outputs = company_model(next(iter(company_dl)).float())
        brand_outputs = brand_model(next(iter(brand_dl)).float())
        category_outputs = category_model(next(iter(category_dl)).float())

        gbl_model_inputs = torch.cat([company_outputs,brand_outputs,category_outputs],dim=1)

        gbl_model_outputs = global_model(gbl_model_inputs)

        preds = gbl_model_outputs.numpy()
    
    ids = pd.read_csv("../data/testHistory.csv").id 
    df = pd.DataFrame(data={"id": ids, "repeatProbability": preds.squeeze()})

    df.to_csv('./eval_test_.csv',index=False)



    
    