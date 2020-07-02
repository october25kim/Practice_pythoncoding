import os
import pandas as pd
import matplotlib.pyplot as plt

import torch

from utils.mypath import data_path, save_path
from utils.args import gae_argparser
from tasks.model_trainer import RNNTrainer



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

            # Save model if validation loss is less than before epoch
            if best_val_loss > val_loss[0]:
                best_val_loss = val_loss[0]

                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))

            # Save train and loss plot
            if epoch % 10 == 0:
                t_loss = sum(train_losses, [])
                v_loss = sum(val_losses, [])

                plt.figure(figsize=(11, 7))
                plt.plot(t_loss, color='cyan', label='Train Loss')
                plt.plot(v_loss, color='red', label='Validation Loss')
                plt.legend()
                plt.savefig(os.path.join(save_path, 'loss_plot.png'))
                plt.close()

    # Save train and validation loss
    pd.Series(train_losses).to_csv(os.path.join(save_path, 'train_loss.csv'))
    pd.Series(val_losses).to_csv(os.path.join(save_path, 'val_loss.csv'))


if __name__ == '__main__':
    main()