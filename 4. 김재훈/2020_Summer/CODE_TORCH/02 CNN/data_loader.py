from torch.utils import data
import dataset as dts

class cifar_loader(object):

    def __init__(self, split, transform, batch_size, shuffle, num_workers):
            
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        root_dir = 'D:/CIFAR10/DATASET'

        split_dataset = dts.CIFAR10(root_dir=root_dir, split=split, transform=transform)
        self.split_loader = data.DataLoader(split_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def loader(self):
        return self.split_loader
