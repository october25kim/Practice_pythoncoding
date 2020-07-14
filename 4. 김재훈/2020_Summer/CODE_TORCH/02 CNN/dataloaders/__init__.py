from dataloaders.dataset import CIFAR
from torch.utils.data import DataLoader

def make_train_loader(args, **kwargs):
    train_set = CIFAR.CIFAR10(args, split='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, shuffle=True, **kwargs)
    return train_loader

def make_val_loader(args, **kwargs):
    val_set = CIFAR.CIFAR10(args, split='val')
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0, shuffle=True, **kwargs)
    return val_loader