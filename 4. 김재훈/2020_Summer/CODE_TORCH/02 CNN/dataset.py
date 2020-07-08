import os
import numpy as np

import torch
import torch.nn as nn

from torch.utils import data
from torchvision import transforms

class CIFAR10(data.Dataset):
    
    def __init__(self, root_dir, split, transform=False):
        super().__init__()
        self.split = split
        self.transform = transform
        self.data_dir = os.path.join(root_dir, self.split)
        self.data_lst = sorted(os.listdir(self.data_dir))

    def __len__(self):
        if self.split:
            return len(self.data_lst)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        img, label = self.pairing(idx)
        pair = {'img':img, 'label':label}

        if self.transform:
            pair = self.transform_totensor(pair)
        
        return pair

    def pairing(self, idx):
        img, label = np.load(self.data_lst[idx]).values()
        return img, label
    
    def transform_totensor(self, pair):
        original_transforms = transforms.Compose([transforms.ToTensor()])
        label_transforms = transforms.Compose([transforms.ToTensor()])

        _img, _label = original_transforms(pair['img']), label_transforms(pair['label'])

        return {'img':_img, 'label':_label}
    