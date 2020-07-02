import os

import torch

from utils.args import gae_argparser
from dataloader import make_data_loader


def main():

    parser = gae_argparser()
    args = parser.parse_args()

    train_loader, val_loader = make_data_loader(args)

    for idx, x in enumerate(train_loader):
        print(len(x))
        data = torch.FloatTensor(x[0])
        data = data.to(device=args.cuda)
        print("___"*30)
        print(idx)
        print("___"*30)
        # print(x.shape)  # torch.Size([64, 4123, 40])
        print("___"*30)
        print(x)  # torch.Size([64, 4123])




if __name__ == '__main__':
    main()
