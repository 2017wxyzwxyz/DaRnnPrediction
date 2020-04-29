from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms as T
#data=pd.read_csv('C:/Users/Administrator/Desktop/train2(3).csv')
#print(data.info())
#print(data.head())
#print(data.dtypes)

class MyDataSet(data.Dataset):
    def __init__(self,csv_file,train=True,test=False,transform=None):
        self.df= pd.read_csv(csv_file)
        self.df['time_stamp']= pd.to_datetime(self.df['time_stamp'])
        self.df=self.df.set_index('time_stamp')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.loc[index]
