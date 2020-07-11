import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import mypath
from torch.utils.data import Dataset
from torchvision import transforms

class CIFAR10(Dataset):
    def __init__(self, args, base_dir=mypath.data_path, split: str = 'train'):
        super().__init__()
        self.args =args
        self.base_dir = base_dir
        self.split = split

        if self.split == 'train':
            self.train_dir = os.path.join(self.base_dir, 'train')
            self.train_folder_list = sorted(os.listdir(os.path.join(self.train_dir)))
            self.train_data_path = [
                '/'.join([self.train_dir, self.train_folder_list[i]])
                for i in range(len(self.train_folder_list))]

        elif self.split == 'val':
            self.val_dir = os.path.join(self.base_dir, 'val')
            self.val_folder_list = sorted(os.listdir(self.val_dir))
            self.val_data_path = [
                '/'.join([self.val_dir, self.val_folder_list[i]])
                for i in range(len(self.val_folder_list))]

        else:
            raise NotImplementedError


    def __getitem__(self, idx):
        img, label = self.make_pair(idx)
        sample = {'image':img, 'label':label}

        if True:#self.args.models == 'ResNet':
            if self.split == 'train':
#                return self.transform_resnet(sample)
                return self.transform_totensor(sample)
            elif self.split == 'val':
#                return self.transform_resnet(sample)
                return self.transform_totensor(sample)
            else:
                NotImplementedError

    def __len__(self):
        if self.split == 'train':
            return len(self.train_folder_list)
        elif self.split == 'val':
            return len(self.val_folder_list)
        else:
            raise NotImplementedError


    def make_pair(self, idx):
        if self.split == 'train':
            img = np.load(os.path.join(self.train_data_path[idx], 'image.npy'))
            label = np.load(os.path.join(self.train_data_path[idx], 'label.npy'))
            return img, label

        elif self.split == 'val':
            img = np.load(os.path.join(self.val_data_path[idx], 'image.npy'))
            label = np.load(os.path.join(self.val_data_path[idx], 'label.npy'))
            return img, label


    def transform_totensor(self, sample):
        original_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        img, label = original_transforms(sample['image']), torch.from_numpy(sample['label']).long()

        return {'image':img, 'label':label}


    # def transform_resnet(self, sample):
    #     RandomCHR_transforms = transforms.Compose([
    #         transforms.RandomCrop(32, padding = 4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomRotation(15)
    #     ])

    #     img, label = RandomCHR_transforms(sample['image']), sample['label']

    #     sample = {'image':img, 'label':label}

    #     return self.transform_totensor(sample)



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    cifar_train = CIFAR10(args, split='train')

    dataloader = DataLoader(cifar_train, batch_size=5, shuffle=True, num_workers=1)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample['image'].size()[0]):
            img = np.transpose(sample['image'],(0,2,3,1))
            gt = sample['label']
            tmp = np.array(img[jj])
            plt.imshow(tmp)

        if ii == 1:
            break
    plt.show(block=True)
