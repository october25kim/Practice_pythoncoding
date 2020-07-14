from dataloader.custom_dataset import CustomDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.args2 import haeun_argparser2
import numpy as np


args = haeun_argparser2()

def make_data_loader(args):

    if args.dataset == "STFT image":
        train_set = CustomDataset(args.feature, split="train")
        test_set = CustomDataset(args.feature, split="test")

        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))

        np.random.seed(args.seed)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)


        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_set, batch_size=args.test_batch_size, sampler=valid_sampler)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


    else:
        raise NotImplementedError
