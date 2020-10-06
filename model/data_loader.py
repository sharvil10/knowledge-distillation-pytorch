"""
   CIFAR-10 data normalization reference:
   https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Normalize, ToTensor, Resize, RandomResizedCrop, RandomRotation, RandomHorizontalFlip, ColorJitter
from torch.utils.data.sampler import SubsetRandomSampler

def fetch_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    mean = [0.5071, 0.4865, 0.4409]
    std_dev = [0.2673, 0.2564, 0.2762]
    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            RandomResizedCrop((64, 64), scale = (0.7, 1.0)),
            RandomRotation(30),
            RandomHorizontalFlip(),
            ColorJitter(0.2, 0.2, 0.2, 0.05),
            ToTensor(),
            Normalize(mean, std_dev)
        ]) 

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([Resize((64, 64)),
                                         ToTensor(),
                                         Normalize(mean, std_dev)])

    # transformer for dev set
    dev_transformer = transforms.Compose([Resize((64, 64)),
                                         ToTensor(),
                                         Normalize(mean, std_dev)])

    trainset = torchvision.datasets.CIFAR100(root='./data-cifar10', train=True,
        download=True, transform=train_transformer)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)

    devset = torchvision.datasets.CIFAR100(root='./data-cifar10', train=False,
        download=True, transform=dev_transformer)
    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl


def fetch_subset_dataloader(types, params):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """

    mean = [0.5071, 0.4865, 0.4409]
    std_dev = [0.2673, 0.2564, 0.2762]
    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            RandomResizedCrop((64, 64), scale = (0.7, 1.0)),
            RandomRotation(30),
            RandomHorizontalFlip(),
            ColorJitter(0.2, 0.2, 0.2, 0.05),
            ToTensor(),
            Normalize(mean, std_dev)
        ]) 

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([Resize((64, 64)),
                                         ToTensor(),
                                         Normalize(mean, std_dev)])

    # transformer for dev set
    dev_transformer = transforms.Compose([Resize((64, 64)),
                                         ToTensor(),
                                         Normalize(mean, std_dev)])

    trainset = torchvision.datasets.CIFAR100(root='./data-cifar10', train=True,
        download=True, transform=train_transformer)

    devset = torchvision.datasets.CIFAR100(root='./data-cifar10', train=False,
        download=True, transform=dev_transformer)

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl
