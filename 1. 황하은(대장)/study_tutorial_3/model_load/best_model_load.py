import tqdm
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


args = haeun_argparser2()

trans = transforms.Compose([
                        transforms.Resize(256),
                        transforms.ToTensor(),
                        ])

test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=trans)
test_dl = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)


loaded_model = torch.load(os.path.join(save_path, 'best_model.pt'))
loaded_model.keys()

model = models.resnet18(num_classes=4)
model.load_state_dict(loaded_model)
model.state_dict()

model = model.to(device=args.cuda)
model.eval()


def test(epoch):

    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        sp_total = 0
        sp_correct = 0
        se_total = 0
        se_correct = 0

    for i, target in enumerate(test_dl):
        inputs, labels = target
        inputs, labels = inputs.to(args.cuda), labels.to(args.cuda)
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)

        if labels.item() == 2:
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

    print(f'Accuracy : {accuracy}')
    print(f'SE : {se}')
    print(f'SP : {sp}')
    print(f'AS : {AS}')

    return accuracy, se, sp, AS

epoch=1
test(epoch)