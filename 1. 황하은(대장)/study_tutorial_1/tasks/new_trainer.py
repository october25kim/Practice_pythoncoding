import tqdm
import torch
import torch.nn as nn
from utils.logger import get_tqdm_config
from dataloader import make_data_loader
from network.RnnModel import LSTMClassifier
from torch.nn import functional as F

import warnings
warnings.filterwarnings(action='ignore')

import easydict
import os
import pandas as pd
import matplotlib.pyplot as plt

from utils.mypath import data_path, save_path
from utils.args import haeun_argparser

args= haeun_argparser()

model = LSTMClassifier(args)
train_loader, val_loader = make_data_loader(args)
for epoch in range(0,60):
    train_loss = 0.0
    train_losses = []
    with tqdm.tqdm(**get_tqdm_config(total=len(train_loader), leave=True, color='green')) as pbar:
        for i, data in enumerate(train_loader):
            x_batch = torch.FloatTensor(data[0])  # torch.Size([16, 4123, 40, 1])
            x_batch = x_batch.squeeze()
            print(x_batch.size())
            y_batch = torch.tensor(data[1])
            print(y_batch.size())  #torch.Size([16, 4123])
            # x_batch = x_batch.to(device=args.cuda)
            # y_batch = y_batch.to(device=args.cuda)
            #

for i, data in enumerate(train_loader):
    x_batch = torch.FloatTensor(data[0])
    x_batch = x_batch.squeeze()
    print(x_batch.size())
    y_batch = torch.tensor(data[1])
    print(y_batch.size())
    x_batch = x_batch.to(device=args.cuda)
    y_batch = y_batch.to(device=args.cuda)
