import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from colorama import Fore
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as opt

from utils.mypath import data_path, ckpt_path, log_path, save_path
from utils.args import resnet_argparser
from dataloaders import make_train_loader, make_val_loader
from networks import ResNet

import warnings
warnings.filterwarnings(action='ignore')

def get_tqdm_config(total, leave=True, color='white'):
    fore_colors = {
        'red': Fore.LIGHTRED_EX,
        'green': Fore.LIGHTGREEN_EX,
        'yellow': Fore.LIGHTYELLOW_EX,
        'blue': Fore.LIGHTBLUE_EX,
        'magenta': Fore.LIGHTMAGENTA_EX,
        'cyan': Fore.LIGHTCYAN_EX,
        'white': Fore.LIGHTWHITE_EX,
    }
    return {
        'file': sys.stdout,
        'total': total,
        'desc': " ",
        'dynamic_ncols': True,
        'bar_format':
            "{l_bar}%s{bar}%s| [{elapsed}<{remaining}, {rate_fmt}{postfix}]" % (fore_colors[color], Fore.RESET),
        'leave': leave
    }


def main():
    pass


class ResNetTrainer(object):
    def __init__(self, args):
        self.args = args


        # Define DataLoader
        self.train_loader = make_train_loader(args)
        self.val_loader = make_val_loader(args)

        # Define model
        self.model = ResNet.resnet50().to(device=self.args.cuda)

        # Define optimizer
        self.optimizer = opt.SGD(self.model.parameters(), lr=args.lr)

        # Define criterion
        self.criterion = nn.CrossEntropyLoss()


    def train(self, epoch):
        train_loss = 0.0
        self.model.train()

        train_losses = []
        with tqdm(**get_tqdm_config(total=len(self.train_loader), leave=True, color='blue')) as pbar:
            for i, sample in enumerate(self.train_loader):
                image = sample['image'].to(device=self.args.cuda)
                label = sample['label'].to(device=self.args.cuda)

                self.optimizer.zero_grad()
                output = self.model(image)

                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                pbar.set_description('Train Loss: %.3f'% (train_loss / (i+1)))
                pbar.update(1)

        train_loss/=(i+1)
        train_losses.append(train_loss)
        print('[EPOCH: %d, numData: %5d]'%(epoch, i*self.args.batch_size + image.data.shape[0]))
        print('Train Loss: %.3f'%train_loss)

        return train_losses

    def validation(self, epoch):
        val_loss = 0.0
        self.model.eval()

        val_losses = []
        with tqdm(**get_tqdm_config(total=len(self.val_loader), leave=True, color='yellow')) as pbar:
            for i, sample in enumerate(self.val_loader):
                image = sample['image'].to(device=self.args.cuda)
                label = sample['label'].to(device=self.args.cuda)

                with torch.no_grad():
                    output = self.model(image)

                loss = self.criterion(output, label)
                val_loss += loss.item()
                pbar.set_description('Validation Loss: %.3f'%(val_loss / (i+1)))
                pbar.update(1)

        val_loss /= (i+1)
        val_losses.append(val_loss)
        print('[EPOCH %d, numData: %5d]'%(epoch, i*self.args.batch_size + image.data.shape[0]))
        print('Validation Loss: %.3f'%val_loss)

        return self.model, val_losses, image, output, label

def main():
    parser = resnet_argparser()
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 0.001

    print(args)
    torch.manual_seed(args.seed)

    trainer = ResNetTrainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epochs:', trainer.args.epochs)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    if not os.path.exists(os.path.join(save_path, 'exp_1')):
        os.makedirs(os.path.join(save_path, 'exp_1'))

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        train_loss = trainer.train(epoch)
        train_losses.append(train_loss)

        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            model, val_loss, image, output, label = trainer.validation(epoch)
            val_losses.append(val_loss)

            # Save model if validation loss is less than before epoch
            if best_val_loss > val_loss[0]:
                best_val_loss = val_loss[0]

                torch.save(model.state_dict(), os.path.join(save_path, 'exp_1', 'best_model.pt'))

            # Save train and loss plot
            if epoch % 10 == 0:
                t_loss = sum(train_losses, [])
                v_loss = sum(val_losses, [])

                plt.figure(figsize=(11, 7))
                plt.plot(t_loss, color='cyan', label='Train Loss')
                plt.plot(v_loss, color='red', label='Validation Loss')
                plt.legend()
                plt.savefig(os.path.join(save_path, 'exp_1', 'loss_plot.png'))
                plt.close()

        # Save train and validation loss
        pd.Series(train_losses).to_csv(os.path.join(save_path, 'exp_1', 'train_loss.csv'))
        pd.Series(val_losses).to_csv(os.path.join(save_path, 'exp_1', 'val_loss.csv'))


if __name__ == '__main__':
    main()