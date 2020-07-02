import os

from utils.mypath import data_path

from dataloader.custom_dataset import LungSoundDataset
import torch.utils.data


def make_data_loader(args):

    if args.dataset == 'stat_mfcc_df':

        train_set = LungSoundDataset(args, csv_file=os.path.join(data_path,'train_stat_mfcc_df.csv'))
        val_set = LungSoundDataset(args, csv_file=os.path.join(data_path,'test_stat_mfcc_df.csv'))


        train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size=16, shuffle=False)

        return train_loader, val_loader
