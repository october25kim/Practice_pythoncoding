import os

import torch

from utils.args2 import haeun_argparser2
from dataloader.custom_dataset import CustomDataset
from dataloader import make_data_loader

def main():
    args = haeun_argparser2()

    train_loader, val_loader, test_loader = make_data_loader(args)

    for i, data in enumerate(train_loader):
        inputs, labels = data
        print(inputs.size())
        print(labels.size())
        print(labels.long().type())
        break




if __name__ == '__main__':
    main()

#
# import torch
# import tqdm
# import torch.nn as nn
# import torch.nn.functional as F
# from utils.args2 import haeun_argparser2
# from utils.logger import get_tqdm_config
# from utils.mypath import data_path, save_path
# import torchvision.models as models
# import os
# import numpy as np
# import pandas as pd
# from utils.mypath import data_path
# import torch
# import torchvision
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torchvision import transforms, utils
# import matplotlib.pyplot as plt
#
#
#
# import warnings
#
# args = haeun_argparser2()
#
# # transform
# trans = transforms.Compose([
#     transforms.Resize(256),
#     transforms.ToTensor(),
# ])
# # Define Dataset
# train_dataset = torchvision.datasets.ImageFolder(root=os.path.join('D:/respiratory_sound project/feature', "train"), transform=trans)
# test_dataset = torchvision.datasets.ImageFolder(root=os.path.join('D:/respiratory_sound project/feature', "test"), transform=trans)
#
# # Define dataloader
# train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
# test_dl = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
#
# for i, data in enumerate(train_dl):
#     inputs, labels = data
#     print(inputs.size())
#     print(labels.size())
#     break