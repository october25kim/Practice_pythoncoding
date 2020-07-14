import tqdm
import torch
import torch.nn as nn
from utils.logger import get_tqdm_config
from dataloader import make_data_loader
from network.RnnModel import LSTMClassifier
from torch.nn import functional as F
import warnings
warnings.filterwarnings(action='ignore')


import os
import pandas as pd
import matplotlib.pyplot as plt

from utils.mypath import data_path, save_path
from utils.args import haeun_argparser





class RNNTrainer(object):
    def __init__(self, args):
        self.args = args

        # Define Data loader
        self.train_loader, self.val_loader = make_data_loader(args)

        # Define model
        model = LSTMClassifier(args)

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Define criterion
        self.criterion = nn.CrossEntropyLoss()

        self.model, self.optimizer = model, optimizer

        # Using cuda
        self.model = self.model.to(device=args.cuda)

    def train(self, epoch):
        train_loss = 0.0
        self.model.train()

        train_losses = []
        with tqdm.tqdm(**get_tqdm_config(total=len(self.train_loader), leave=True, color='green')) as pbar:
            for i, data in enumerate(self.train_loader):
                x_batch= torch.FloatTensor(data[0])
                x_batch = x_batch.squeeze()
                y_batch =  torch.tensor(data[1], dtype=torch.long)
                x_batch = x_batch.to(device=self.args.cuda)
                y_batch = y_batch.to(device=self.args.cuda)


                self.optimizer.zero_grad()
                out = self.model(x_batch)
                loss = self.criterion(out, torch.max(y_batch, 1)[1])
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                pbar.set_description('Train Loss: %.3f' % (train_loss / (i + 1)))
                pbar.update(1)

        train_loss /= (i + 1)

        train_losses.append(train_loss)
        print('[Epoch: %d, numData: %5d]' % (epoch, i * self.args.batch_size + data[0].shape[0]))
        print('Train Loss: %.3f' % train_loss)

        return train_losses

    def validation(self, epoch):
        self.model.eval()
        correct, total = 0, 0
        val_losses =[]
        val_loss = 0.0

        with tqdm.tqdm(**get_tqdm_config(total=len(self.val_loader), leave=True, color='yellow')) as pbar:
            for i, data in enumerate(self.val_loader):
                x_batch= torch.FloatTensor(data[0])
                x_batch = x_batch.squeeze()
                y_batch = torch.tensor(data[1], dtype=torch.long)
                x_batch = x_batch.to(device=self.args.cuda)
                y_batch = y_batch.to(device=self.args.cuda)


                with torch.no_grad():
                    out = self.model(x_batch)
                    loss = self.criterion(out, torch.max(y_batch, 1)[1])
                    val_loss += loss.item()

                    # 가장 높은 값을 가진 인덱스가 바로 예측값
                    preds = F.log_softmax(out, dim=1).argmax(dim=1)
                    total += y_batch.size(0)
                    correct += (preds == y_batch).sum().item()

                acc = correct / total
                val_loss /= len(self.val_loader.dataset)
                val_losses.append(val_loss)
                test_accuracy = 100. * correct / len(self.val_loader.dataset)
                print('[Epoch: %d, numData: %5d]' % (epoch, i * self.args.batch_size + data[0].shape[0]))
                print('Train Loss: %.3f' % val_loss)

        return self.model, val_losses


def main():
    parser = gae_argparser()
    args = parser.parse_args()

    print(args)
    torch.manual_seed(args.seed)

    trainer = RNNTrainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epochs:', trainer.args.epochs)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        train_loss = trainer.train(epoch)
        train_losses.append(train_loss)
        print('finish train')

        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            print('start validation')
            model, val_loss = trainer.validation(epoch)
            val_losses.append(val_loss)


if __name__ == '__main__':
    main()