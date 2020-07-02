import os
import numpy as np
import pandas as pd
from utils.mypath import data_path
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class LungSoundDataset(object):
    def __init__(self, args, csv_file, base_dir=data_path, transform=None):

        self.base_dir = base_dir
        self.mfcc_data = pd.read_csv(csv_file)
        self.mfcc_data = self.mfcc_data.drop("Unnamed: 0", axis=1)
        col = self.mfcc_data.columns[:40]
        self.X = np.array(self.mfcc_data[col])
        self.y = np.array(self.mfcc_data['encoding_class'])


        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.mfcc_data)

    def __getitem__(self, idx):
        X_data = torch.FloatTensor(np.array(self.X))
        X_data = X_data.unsqueeze(2)
        y_data = torch.FloatTensor(np.array(self.y))
        # Dataset = {"X": X_data,
        #            "y": y_data}




        # if self.transform:
        #     Dataset = self.transform(Dataset)

        return X_data, y_data





# base_dir = data_path
# csv_file=os.path.join(data_path,'train_stat_mfcc_df.csv')
# mfcc_data = pd.read_csv(csv_file)
# mfcc_data = mfcc_data.drop("Unnamed: 0", axis=1)
# col = mfcc_data.columns[:40]
# X = np.array(mfcc_data[col])
# y = np.array(mfcc_data['encoding_class'])
# y_data = torch.FloatTensor(np.array(y))
# X_data = torch.FloatTensor(X)
# X_data=X_data.unsqueeze(2)
# X_data
# y_data.size()
# Dataset = TensorDataset(X_data, y_data)
# Dataset
#
# train_loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=64, shuffle=True)
#
# for idx, x in enumerate(train_loader):
#     print("___" * 30)
#     print(idx)
#     print("___" * 30)
#     # print(x.size())  # torch.Size([64, 4123, 40])
#     print("___" * 30)
#     print(x)  # torch.Size([64, 4123])