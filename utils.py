from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset,random_split
import torch
import pandas as pd
import numpy as np 

pd.set_option('display.max_columns', None)
torch.manual_seed(0)
class ShoppersDataset(Dataset):
    def __init__(self,X):
        self.X = X
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index,:]

def load_datasets(data_path,batch_size):
    df = pd.read_csv(data_path)

    targets = df['repeater'].values.reshape((-1,1))
    df = df.drop('repeater',axis=1)

    comp_cols, brand_cols, cat_cols = [],[],[]
    cpc,brc,ctc = 0,0,0
    for i,col in enumerate(df.columns):
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
    
    comp_ds = df[comp_cols].copy()
    cat_ds = df[cat_cols].copy()
    brand_ds = df[brand_cols].copy()
    
    comp_dl = DataLoader(
        dataset=ShoppersDataset(df[comp_cols].copy().to_numpy()),
        batch_size=batch_size,
        shuffle=False
    )
    cat_dl = DataLoader(
        dataset=ShoppersDataset(df[cat_cols].copy().to_numpy()),
        batch_size=batch_size,
        shuffle=False
    )
    brand_dl = DataLoader(
        dataset=ShoppersDataset(df[brand_cols].copy().to_numpy()),
        batch_size=batch_size,
        shuffle=False
    )

    return {'data': [comp_dl,cat_dl,brand_dl], 'labels': targets}

if __name__ == "__main__":
    d = load_datasets("../data/train_data.csv", batch_size=64)
    import pickle
    with open('data.pt','wb') as outfile:
        pickle.dump(d, outfile)
    
