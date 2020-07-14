import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils.args2 import haeun_argparser2
from utils.logger import get_tqdm_config
from utils.mypath import data_path, save_path
import torchvision.models as models
import os
import numpy as np
import pandas as pd
from utils.mypath import data_path
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from dataloader import make_data_loader
import warnings


warnings.filterwarnings(action='ignore')


class ResNetTrainer(object):
    def __init__(self, args):
        self.args = haeun_argparser2()

        # Define dataloader
        self.train_dl, self.val_dl, self.test_dl = make_data_loader(args)

        # Define model
        self.model = models.resnet18(num_classes=4)

        # Define optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)

        # Define criterion
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weight_func())
        # self.criterion = nn.CrossEntropyLoss()
        # Using cuda
        self.model = self.model.to(device=args.cuda)

        # Dataset length
        self.train_total_batch, self.val_total_batch, self.test_total_batch = \
            len(self.train_dl), len(self.val_dl), len(self.test_dl)

    def train(self, epoch):
        trn_loss = 0.0
        self.model.train()

        train_losses = []
        with tqdm.tqdm(**get_tqdm_config(total=len(self.train_dl), leave=True, color='green')) as pbar:
            for i, data in enumerate(self.train_dl):
                inputs, labels = data


                inputs, labels = inputs.to(self.args.cuda), labels.to(self.args.cuda)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                labels = labels.squeeze(1)
                labels = labels.long()


                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                trn_loss += loss.item()
                pbar.set_description('Train Loss: %.3f' % (trn_loss / (i + 1)))
                pbar.update(1)

        trn_loss /= self.train_total_batch  # 한 에폭에 대한 loss

        train_losses.append(trn_loss)
        print('[Epoch: %d, numData: %5d]' % (epoch, self.train_total_batch))
        print('Train Loss: %.3f' % trn_loss)

        return trn_loss

    def valid(self, epoch):
        valid_loss = 0.0
        self.model.eval()

        valid_losses = []
        with tqdm.tqdm(**get_tqdm_config(total=len(self.val_dl), leave=True, color='red')) as pbar:

                with torch.no_grad():
                    correct = 0
                    total = 0
                    sp_total = 0
                    sp_correct = 0
                    se_total = 0
                    se_correct = 0

                    for i, target in enumerate(self.val_dl):
                        inputs, labels = target
                        # labels = labels.squeeze(1)
                        inputs, labels = inputs.to(self.args.cuda), labels.to(self.args.cuda)
                        outputs = self.model(inputs)
                        labels = labels.squeeze(1)
                        labels = labels.long()

                        loss = self.criterion(outputs, labels)
                        valid_loss += loss.item()
                        pbar.set_description('Validation Loss: %.3f' % (valid_loss / (i + 1)))
                        pbar.update(1)

                        _, predicted = torch.max(outputs.data, 1)

                        if labels.item() == 0:
                            sp_total += 1
                            sp_correct += (predicted == labels).sum().item()

                        else:
                            se_total += 1
                            se_correct += (predicted == labels).sum().item()

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()


        valid_loss /= self.val_total_batch

        valid_losses.append(valid_loss)
        print('[Epoch: %d, numData: %5d]' % (epoch, self.val_total_batch))
        print('Valid Loss: %.3f' % valid_loss)

        accuracy = correct / total
        se = se_correct / se_total
        sp = sp_correct / sp_total

        AS = (se + sp) / 2

        return self.model, valid_loss, accuracy, se, sp, AS

    def test(self, epoch):
        test_loss = 0.0
        self.model.eval()

        test_losses = []
        with tqdm.tqdm(**get_tqdm_config(total=len(self.test_dl), leave=True, color='blue')) as pbar:

                with torch.no_grad():
                    correct = 0
                    total = 0
                    sp_total = 0
                    sp_correct = 0
                    se_total = 0
                    se_correct = 0

                    for i, target in enumerate(self.test_dl):
                        inputs, labels = target
                        # labels = labels.squeeze(1)
                        inputs, labels = inputs.to(self.args.cuda), labels.to(self.args.cuda)
                        outputs = self.model(inputs)
                        labels = labels.squeeze(1)
                        labels = labels.long()

                        _, predicted = torch.max(outputs.data, 1)
                        pbar.set_description(f'Test')
                        pbar.update(1)

                        if labels.item() == 0:
                            sp_total += 1
                            sp_correct += (predicted == labels).sum().item()

                        else:
                            se_total += 1
                            se_correct += (predicted == labels).sum().item()

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

        accuracy = correct / total
        se = se_correct / se_total
        sp = sp_correct / sp_total

        AS = (se + sp) / 2

        return self.model, accuracy, se, sp, AS




    def class_weight_func(self):
        weight_class = []

        data = pd.read_csv('Z:/1. 프로젝트/2020_COVID/COVID_SOUND/data/master_df.csv').drop("Unnamed: 0",axis=1)

        for i in range(4):
            weight_class.append((1 / data['class'].value_counts()[i]) * 100)

        class_weights = F.softmax(torch.tensor(weight_class, dtype=torch.float)).cuda()

        return class_weights


