from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset,random_split
import torch
import pandas as pd
import numpy as np 
from pickle import dump

pd.set_option('display.max_columns', None)
torch.manual_seed(0)
class ShoppersDataset(Dataset):
    def __init__(self,X):
        self.X = X
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index,:]

def create_train_test(df,cols, train_index, test_index, batch_size):

    train_ds = ShoppersDataset(
        df.loc[train_index][cols].copy().to_numpy()
    )
    test_ds = ShoppersDataset(
        df.loc[test_index][cols].copy().to_numpy()
    )
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)
    return train_dl,test_dl

def load_datasets(data_path,batch_size):
    df = pd.read_csv(data_path)

    targets = df['repeater']

    train_index, test_index = train_test_split(df.index.values,test_size=0.1,shuffle=False)
    # treat labels as dataloaders
    train_labels_1, test_labels_1 = create_train_test(
        df=df, cols=['repeater'], train_index=train_index, test_index=test_index, batch_size=batch_size
    )

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
    
    

    train_comp_dl, test_comp_dl = create_train_test(df, cols=comp_cols,  train_index=train_index, test_index=test_index, batch_size=batch_size)
    train_cat_dl, test_cat_dl = create_train_test(  df, cols=cat_cols,   train_index=train_index, test_index=test_index, batch_size=batch_size)
    train_brand_dl, test_brand_dl = create_train_test( df, cols=brand_cols, train_index=train_index, test_index=test_index, batch_size=batch_size)

    
    train_labels = targets.loc[train_index].values.reshape((-1,1))
    test_labels  = targets.loc[test_index ].values.reshape((-1,1))


    return {
        'data': {
            'train': [train_comp_dl,train_cat_dl,train_brand_dl], 
            'test': [test_comp_dl,test_cat_dl,test_brand_dl]
        },    
        'train_labels': train_labels, 'test_labels': test_labels, 
        'batch_size': batch_size
    }

def save_data(data_path,batch_size,outfile):
    data = load_datasets(
        data_path=data_path, 
        batch_size=batch_size
    )
    with open(outfile,'wb') as f:
        dump(data, f)
    
    return data

if __name__ == "__main__":
    BATCH_SIZE = 32
    outfile = './data.pt'

    DATA = save_data(data_path='../data/train_data.csv', batch_size=BATCH_SIZE,outfile=outfile)
    
    pass