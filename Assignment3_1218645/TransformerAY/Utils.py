import numpy as np
import torch
from MyNLPDataSet import MyNLPDataSet
from torch.utils.data import DataLoader
import gzip

def cycle(loader):
    while True:
        for data in loader:
            yield data

def get_loaders_enwiki8(seq_len, batch_size):
    with gzip.open('C:/Users/ayann/Desktop/Anand/Assignment3_1218645/Assignment3_1218645/enwik8.gz') as file:
        data = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        data_train, data_val = map(torch.from_numpy, np.split(data, [int(90e6)]))
        train_datset = MyNLPDataSet(data_train, seq_len)
        val_datset =MyNLPDataSet(data_val, seq_len)
        train_Loader = cycle(DataLoader(train_datset,batch_size=batch_size))
        val_loader = cycle(DataLoader(val_datset,batch_size=batch_size))
        return train_Loader , val_loader , val_datset
    
    