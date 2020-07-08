import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from data_loader import cifar_loader
from network.models import resnet50

import warnings
warnings.filterwarnings(action='ignore')


class CNNTrainer(object):
    def __init__(self):
        # args
        self.batch_size = 32
        # Define CUDA
        USE_CUDA = torch.cuda.is_available()
        DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

        #Define Data Loader
        self.train_loader = cifar_loader(split='train', transform=True, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_loader = cifar_loader(split='val', transform=True, batch_size=self.batch_size, shuffle=True, num_workers=2)

        # Define Model
        self.model = resnet50().to(DEVICE)

        # Define Optimizer
        lr = 0.01
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr, momentum = 0.9)

        # Define Criterion
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epoch):
        train_loss = 0.0
        train_losses = []
        self.model.train()
        with tqdm.tqdm(total=len(self.train_loader)) as tbar:
            for batch_idx, sample in enumerate(self.train_loader):
                img = sample['img'].to(DEVICE)
                label = sample['label'].to(DEVICE)

                self.optimizer.zero_grad()
                output = self.model(img)

                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                tbar.set_description('Train loss: %.3f'%(train_loss/(batch_idx+1)))
                tbar.update(1)
        train_loss /= (batch_idx + 1)
        train_losses.append(train_loss)
        print('[Epoch: %d, numImages: %5d]' % (epoch, batch_idx * self.batch_size + img.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        return train_losses

    
    def validation(self, epoch):
        self.model.eval()

        eval_loss = 0.0
        eval_losses = []
        correct = 0.0
        
        with tqdm.tqdm(total=len(self.val_loader)) as tbar:
            for batch_idx, sample in enumerate(self.val_loader):
                img = sample['img'].to(DEVICE)
                label = sample['label'].to(DEVICE)

                with torch.no_grad():
                    output = self.model(img)

                loss = self.criterion(output, label)

                eval_loss += loss.item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()

                tbar.set_description('Test loss: %.3f'%(eval_loss/(batch_idx+1)))
                tbar.update(1)
        eval_loss /= (batch_idx + 1)
        eval_losses.append(eval_loss)
        print('[Epoch: %d, numImages: %5d]' % (epoch, batch_idx * self.batch_size + img.data.shape[0]))
        print('Loss: %.3f' % eval_loss)
        print('ACC: %.3f' % (100. * correct / len(self.val_loader)))

        return eval_losses







        
        