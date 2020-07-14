import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args2 import haeun_argparser2
from utils.logger import get_tqdm_config
from utils.mypath import data_path, save_path
from task.trainer import ResNetTrainer
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
from torch.utils.tensorboard import SummaryWriter

if not os.path.exists(os.path.join(save_path, 'SGD_weight/tensorboard_loss_plot/')):
    os.makedirs(os.path.join('SGD_weight/tensorboard_loss_plot/'))

writer = SummaryWriter(os.path.join('SGD_weight/tensorboard_loss_plot/'))



def main():     
    args = haeun_argparser2()

    print(args)
    torch.manual_seed(args.seed)

    trainer = ResNetTrainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epochs:', trainer.args.epochs)

    best_test_loss = float('inf')
    train_losses, valid_losses = [], []
    best_AS = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        train_loss = trainer.train(epoch)
        train_losses.append(train_loss)


        model, valid_loss, accuracy, se, sp, AS = trainer.valid(epoch)
        valid_losses.append(valid_loss)

        test_model, test_accuracy, test_se, test_sp, test_AS = trainer.test(epoch)

        torch.save(model.state_dict(),
                   save_path + str(epoch) + "_sp_" + str(round(100 * sp, 3)) +
                   "_se_" + str(round(100 * se, 3)) +
                   "_as_" + str(round(100 * AS, 3)) + ".pt")

        # Save model if validation loss is less than before epoch
        if test_AS * 100 > best_AS:
            if not os.path.exists(os.path.join(save_path, "best_model")):
                os.makedirs(os.path.join(save_path, "best_model"))

            torch.save(model.state_dict(),
                       save_path + 'best_model/'+ str(epoch) + "_sp_" + str(round(100 * test_sp, 3)) +
                            "_se_" + str(round(100 * test_se, 3)) +
                            "_as_" + str(round(100 * test_AS, 3)) + ".pt")
            best_AS = test_AS * 100
            print(best_AS)


        # plt.figure(figsize=(11, 7))
        # plt.plot(train_losses, color='cyan', label='Train Loss')
        # plt.plot(valid_losses, color='red', label='Validation Loss')
        # plt.legend()
        # plt.savefig(os.path.join(save_path, 'loss_plot.png'))
        # plt.close()

        writer.add_scalars('loss', {'_train': train_loss,
                                    '_valid': valid_loss}, epoch)

        writer.add_scalars('valid_score', {'_SP': 100 * sp,
                                            '_SE': 100 * se,
                                            '_AS': 100 * AS}, epoch)

        writer.add_scalars('test_score', {'_SP': 100 * test_sp,
                                          '_SE': 100 * test_se,
                                         '_AS': 100 * test_AS}, epoch)

    # Save train and validation loss
    pd.Series(train_losses).to_csv(os.path.join(save_path, 'train_loss.csv'))
    pd.Series(valid_losses).to_csv(os.path.join(save_path, 'valid_loss.csv'))


if __name__ == '__main__':
    main()

