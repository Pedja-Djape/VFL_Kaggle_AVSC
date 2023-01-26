from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset,random_split
import torch
import pandas as pd
import numpy as np 
from pickle import dump


class ShoppersDataset(Dataset):
    """
        The class used to represent a given client's data set in VFL (no labels).

        Attributes
        __________
        X: numpy.ndarray
            A numppy array representing client's dataset. Each row represents a consumer's features.
        """
    def __init__(self,X):
        self.X = X
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index,:]

def create_train_test(df,cols, train_index, test_index, batch_size):
    """
    Create train and test DataLoader objects using specified parameters.

    Args:
        df: pandas.DataFrame
            DataFrame object containing data to be used for training and testing purposes.
        cols: list
            A subset of columns of `df` to include when creating train/test DataLoaders.
        train_index: array-like
            An iterable that contains indices representing training samples in `df`. 
        test_index: array-like
            An iterable that contains indices representing testing samples in `df`. 
        batch_size: int
            Batch size to use while training/testing models.
    
    Returns:
        tuple(DataLoader)
            A tuple that includes the train and test DataLoaders. 
    """

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
    """
    Creates vertically-split datasets to use for vertical federated learning.

    Args:
        data_path: str
            Path to the .csv file that includes data for training and testing.
        batch_size: int
            Batch size to use while training/testing models.
    
    Returns:
        dict
            A dictionary that contains the vertically-split datasets for use in federated learning along 
            with associated train/test labels. 
    """
    df = pd.read_csv(data_path)

    # Get train/test indices
    train_index, test_index = train_test_split(df.index.values,test_size=0.1,shuffle=False)
    # treating labels as dataloaders --> simplifies VFL training.
    # get labels
    train_labels, test_labels = create_train_test(
        df=df, cols=['repeater'], train_index=train_index, test_index=test_index, batch_size=batch_size
    )
    # drop repeater column (clients don't have labels)
    df = df.drop('repeater',axis=1)

    # Split's data into three types of clients: brand-,category-, and company-based.
    comp_cols, brand_cols, cat_cols = [],[],[]
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
    
    
    # create dataloaders for each client
    train_comp_dl, test_comp_dl = create_train_test(df, cols=comp_cols,  train_index=train_index, test_index=test_index, batch_size=batch_size)
    train_cat_dl, test_cat_dl = create_train_test(  df, cols=cat_cols,   train_index=train_index, test_index=test_index, batch_size=batch_size)
    train_brand_dl, test_brand_dl = create_train_test( df, cols=brand_cols, train_index=train_index, test_index=test_index, batch_size=batch_size)

    rval = {
        'data': {
            'company': {'train': train_comp_dl, 'test': test_comp_dl},
            'brand': {'train': train_brand_dl, 'test': test_brand_dl},
            'category': {'train': train_cat_dl, 'test': test_cat_dl}
        },
        'train_labels': train_labels, 'test_labels': test_labels, 
        'batch_size': batch_size
    }
    # rval = {
    #     'data': {
    #         'train': [train_comp_dl,train_cat_dl,train_brand_dl], 
    #         'test': [test_comp_dl,test_cat_dl,test_brand_dl]
    #     },    
    #     'train_labels': train_labels, 'test_labels': test_labels, 
    #     'batch_size': batch_size
    # }
    return rval 

def save_data(data_path,batch_size,outfile):
    """
    Creates client datasets and saves to pickled file for future use.

    Args:
        data_path: str
            Path to the .csv file that includes data for training and testing.
        batch_size: int
            Batch size to use while training/testing models.
        outfile: str
            Output file destination of pickled dataset dict. 
    
    Returns:
        dict
            A dictionary that contains the vertically-split datasets for use in federated learning along 
            with associated train/test labels. 
    """
    data = load_datasets(
        data_path=data_path, 
        batch_size=batch_size
    )
    with open(outfile,'wb') as f:
        dump(data, f)
    
    return data

class ClientIdentifier:

    def __init__(self):
        self.cid_to_client_map = {
            0: 'company',
            1: 'category',
            2: 'brand'
        }
        self.client_to_cid = {
            'company': 0,
            'category': 1,
            'brand': 2
        }
    def get_cid_from_client(self,client_type):
        return self.client_to_cid[client_type]
    
    def get_client_from_cid(self,cid):
        return self.cid_to_client_map[cid]

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    torch.manual_seed(0)

    # set batch size of 32
    BATCH_SIZE = 32
    outfile = './data.pt'

    DATA = save_data(data_path='../data/train_data.csv', batch_size=BATCH_SIZE,outfile=outfile)
    pass