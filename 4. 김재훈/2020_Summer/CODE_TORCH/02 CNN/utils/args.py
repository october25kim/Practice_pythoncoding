import argparse
import torch

def resnet_argparser():
    parser = argparse.ArgumentParser(description='Pytorch ResNet Training')

    # dataloader
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Validation batch size')

    # ResNet network parameters
    parser.add_argument('--spec_dim', type=int, default=5,
                        help='spec hidden dimension')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch number')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='evaluation interval (default:1')
    parser.add_argument('--no_val', action='store_true', default=False,
                        help='skip validation during training')


    # Training & validation parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default=auto)')

    # cuda & seed
    parser.add_argument('--cuda', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Using gpu')
    parser.add_argument('--seed', type=int, default=2020010553,
                        help='random seed (default: 123)')

    return parser