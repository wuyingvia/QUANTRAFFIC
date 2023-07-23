import pickle
import torch
import numpy as np
import pandas as pd

def masked_mae(preds, labels, null_val=0.0):
    preds[preds<1e-5]=0
    labels[labels<1e-5]=0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def inverse_transform(x, mean, std):
    return x * std + mean

def transform(x, mean, std):
    return (x - mean) / std

def load_data(DATANAME, FLOWPATH):
    if DATANAME in['PEMS04', 'PEMS08']:
        data = np.squeeze(np.load(FLOWPATH)['data'])[..., 0]
    elif DATANAME in ['PEMS03','PEMS07']:
        data = np.squeeze(np.load(FLOWPATH)['data'])
    elif DATANAME in ['METR-LA', 'PEMS-BAY']:
        data = pd.read_hdf(FLOWPATH).values
    elif DATANAME == 'PEMSD7M':
        data = pd.read_csv(FLOWPATH, index_col=[0]).values
    else:
        raise ValueError(f"Invalid DATANAME: {DATANAME}")
    return data
