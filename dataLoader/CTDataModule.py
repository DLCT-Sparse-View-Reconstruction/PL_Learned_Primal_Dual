import os

import torch

from pytorch_lightning import LightningDataModule

import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split

class CTDataModule(LightningDataModule):
    def __init__(self, data_dir: str, vl_data_dir: str, ts_data_dir: str, batch_size: int):
        super(CTDataModule, self).__init__()
        self.data_dir = data_dir
        self.vl_data_dir = vl_data_dir
        self.ts_data_dir = ts_data_dir
        self.batch_size = batch_size
        
        self.num_workers = max(int(os.cpu_count()), 1)

    def setup(self, stage=None):
      # download
      # load the numpy data array
        print("Train -- BEGIN__Loading the dataset__")
        with open(self.data_dir, 'rb') as f:
            dataset = torch.Tensor(np.load(f))
        print("Train -- DONE__Loading the dataset__") 
        
        if self.ts_data_dir != None and self.vl_data_dir != None:
            print("Valid -- BEGIN__Loading the dataset__")
            with open(self.vl_data_dir, 'rb') as f:
                valid_dataset = torch.Tensor(np.load(f))
            print("Valid -- DONE__Loading the dataset__")   

            print("Test -- BEGIN__Loading the dataset__")
            with open(self.ts_data_dir, 'rb') as f:
                test_dataset = torch.Tensor(np.load(f))
            print("Test -- DONE__Loading the dataset__")         

        dataset_size = len(dataset)
        train_size = int(0.9*dataset_size)
        val_size = int(0.1*dataset_size)

        if self.ts_data_dir == None:
            test_size = dataset_size - train_size - val_size

            self.train_set, self.val_set, self.test_set = random_split(dataset[:dataset_size], [train_size, val_size, test_size])
        else:
            self.train_set, self.val_set, self.test_set = dataset[:1], valid_dataset[:1], test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle = True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)
