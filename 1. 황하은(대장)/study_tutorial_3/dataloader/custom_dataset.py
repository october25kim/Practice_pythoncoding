import os
import numpy as np
import pandas as pd
from utils.mypath import TOP_DIR, data_path
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
from utils.args2 import haeun_argparser2


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

args = haeun_argparser2()


class CustomDataset(Dataset):
    def __init__(self, feature, split="train"):
        self.split = split
        self.base_dir = "D:/respiratory_sound project/feature/"
        self.feature = feature
        self.train_datalist = os.listdir(os.path.join(self.base_dir + self.feature, 'train/array_with_label'))
        self.test_datalist = os.listdir(os.path.join(self.base_dir + self.feature, 'test/array_with_label'))

    # 총 데이터의 개수를 리턴
    def __len__(self):
        if self.split == "train":
            return len(self.train_datalist)
        elif self.split == "test":
            return len(self.test_datalist)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        if self.split == "train":
            sample = np.load(os.path.join(self.base_dir + self.feature + "/train/array_with_label", self.train_datalist[idx]))
            sample = {
                'img': sample['img'],
                'label': sample['label']
            }
        elif self.split == "test":
            sample = np.load(os.path.join(self.base_dir + self.feature + "/test/array_with_label", self.test_datalist[idx]))
            sample = {
                'img': sample['img'],
                'label': sample['label']
            }

        return self.transform_totensor(sample)

    def transform_totensor(self, sample):
        input_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        _img, _label = input_transforms(sample['img']), torch.from_numpy(sample['label'])

        return _img.float(), _label.float()

