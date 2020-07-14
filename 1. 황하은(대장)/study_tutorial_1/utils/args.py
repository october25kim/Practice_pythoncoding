import argparse
import easydict

# def gae_argparser():
#     parser = argparse.ArgumentParser(description='Lung sound RNN example')
#
#     # dataloader
#     parser.add_argument('--dataset', type=str, default='stat_mfcc_df',
#                         help='dataset')
#     #
#     # Model
#     parser.add_argument('--input_dim', type=int, default=40,
#                         help='features (X) dimension')
#     parser.add_argument('--hidden_dim', type=int, default=256,
#                         help='hidden layer dimension')
#     parser.add_argument('--layer_dim', type=int, default=1,
#                         help='num of layer ')
#     parser.add_argument('--output_dim', type=int, default=4,
#                         help='output dimension ')
#
#
#     # optimizer
#     parser.add_argument('--lr', type=float, default=1e-4,
#                         help='optimizer learning rate (default: auto)')
#
#     # cuda available & seed
#     parser.add_argument('--cuda', type=str, default='cuda:0',
#                         help='Using cuda (default: cuda:0)')
#     parser.add_argument('--seed', type=int, default=123,
#                         help='seed number')
#
#     # training hyperparameters
#     parser.add_argument('--epochs', type=int, default=60,
#                         help='Number of epochs to train (default: auto)')
#     parser.add_argument('--start_epoch', type=int, default=0,
#                         help='start epoch number (default: 0)')
#     parser.add_argument('--batch_size', type=int, default=64,
#                         help='input batch size for training (default: auto)')
#     parser.add_argument('--val_batch_size', type=int, default=64,
#                         help='input batch size for validation (default: auto)')
#
#     # Validation step
#     parser.add_argument('--eval_interval', type=int, default=1,
#                         help='evaluation interval (default: 1)')
#     parser.add_argument('--no_val', action='store_true', default=False,
#                         help='skip validation during training')
#
#     return parser


def haeun_argparser():
    args = easydict.EasyDict({
        "dataset": "stat_mfcc_df",
        "input_dim": 40,
        "hidden_dim": 256,
        "layer_dim": 1,
        "output_dim": 4,
        "lr": 1e-4,
        "cuda": "cuda:0",
        "seed": 123,
        "epochs": 60,
        "start_epoch": 0,
        "batch_size": 64,
        "val_batch_size": 64,
        "eval_interval": 1,
        "no_val": False
    })
    return args