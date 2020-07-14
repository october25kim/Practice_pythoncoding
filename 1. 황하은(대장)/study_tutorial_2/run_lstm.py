import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.logger import get_tqdm_config
from utils.mypath import data_path, save_path
from utils.args import haeun_argparser
from torch.nn import functional as F
from network.RnnModel import LSTMClassifier
from dataloader.custom_dataloader import create_datasets, create_loaders


args= haeun_argparser()

ROOT = "D:/tutorial data/career-con-2019/"
SAMPLE = os.path.join(ROOT,'sample_submission.csv')
TRAIN = os.path.join(ROOT,'X_train.csv')
TARGET = os.path.join(ROOT,'y_train.csv')
TEST = os.path.join(ROOT,'X_test.csv')


ID_COLS = ['series_id', 'measurement_number']

x_cols = {
    'series_id': np.uint32,
    'measurement_number': np.uint32,
    'orientation_X': np.float32,
    'orientation_Y': np.float32,
    'orientation_Z': np.float32,
    'orientation_W': np.float32,
    'angular_velocity_X': np.float32,
    'angular_velocity_Y': np.float32,
    'angular_velocity_Z': np.float32,
    'linear_acceleration_X': np.float32,
    'linear_acceleration_Y': np.float32,
    'linear_acceleration_Z': np.float32
}

y_cols = {
    'series_id': np.uint32,
    'group_id': np.uint32,
    'surface': str
}

x_trn = pd.read_csv(TRAIN, usecols=x_cols.keys(), dtype=x_cols)
x_tst = pd.read_csv(TEST, usecols=x_cols.keys(), dtype=x_cols)
y_trn = pd.read_csv(TARGET, usecols=y_cols.keys(), dtype=y_cols)

print('Preparing datasets')
trn_ds, val_ds, enc = create_datasets(x_trn, y_trn['surface'])

train_loader, val_loader = create_loaders(trn_ds, val_ds)

#Define model
model = LSTMClassifier(args)

#Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#Define criterion
criterion = nn.CrossEntropyLoss()

#Using cuda
model = model.to(device=args.cuda)

print("start training")
best_acc = 0

for epoch in range(1, args.epochs+ 1):

    for i, (x_batch, y_batch) in enumerate(train_loader):
        model.train()
        # print(f'x_batch_size: {x_batch.size()}')  #[64, 128, 10]  (배치사이즈, ,input_dim)
        # print(f'y_batch_size: {y_batch.size()}')  #[64]
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0
    print("start val")
    for x_val, y_val in val_loader:
        print(f'x_batch_size: {x_val.size()}')  #[64, 128, 10] (배치사이즈, ,input_dim)
        print(f'x_batch_size: {y_val.size()}')  #[64]

        x_val, y_val = [t.cuda() for t in (x_val, y_val)]


        out = model(x_val)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_val.size(0)
        correct += (preds == y_val).sum().item()

    acc = correct / total

    if epoch % 5 == 0:
        print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')

    if acc > best_acc:
        trials = 0
        best_acc = acc
        torch.save(model.state_dict(), 'best.pth')
        print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
    else:
        trials += 1
        if trials >= args.patience:
            print(f'Early stopping on epoch {epoch}')
            break