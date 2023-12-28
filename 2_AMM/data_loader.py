import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, SVHN, MNIST, EMNIST, KMNIST, FashionMNIST
from torchvision import transforms
import numpy as np
import json

COLOR = ['c10', 'c100', 'in', 's']
BW = ['m', 'em', 'km', 'fm']

def get_data(root, dataset, val_split=0.5):
    if dataset in COLOR:
        if dataset.lower() == 'in':
            INtrain_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            INtest_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
    elif dataset in BW:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    if dataset.lower() == 'c10':
        dataset_class = CIFAR10
    elif dataset.lower() == 'c100':
        dataset_class = CIFAR100
    elif dataset.lower() == 'in':
        dataset_class = ImageNet
    elif dataset.lower() == 's': 
        dataset_class = SVHN
    elif dataset.lower() == 'm':
        dataset_class = MNIST
    elif dataset.lower() == 'em':
        dataset_class = EMNIST
    elif dataset.lower() == 'km':
        dataset_class = KMNIST
    elif dataset.lower() == 'fm':
        dataset_class = FashionMNIST

    if dataset_class == MNIST or dataset_class == KMNIST or dataset_class == FashionMNIST:
        full_train_dataset = dataset_class(root=root,
                                           train=True,
                                           download=True,
                                           transform=transform)
        test_dataset = dataset_class(root=root,
                                     train=False,
                                     download=True,
                                     transform=transform)
    if dataset_class == EMNIST:
        full_train_dataset = dataset_class(root=root,
                                           split='byclass',
                                           train=True,
                                           download=True,
                                           transform=transform)
        test_dataset = dataset_class(root=root,
                                     split='byclass',
                                     train=False,
                                     download=True,
                                     transform=transform)
        
    if dataset_class == CIFAR10 or dataset_class == CIFAR100:
        full_train_dataset = dataset_class(root=root,
                                           train=True,
                                           download=True,
                                           transform=train_transform)
        test_dataset = dataset_class(root=root,
                                     train=False,
                                     download=True,
                                     transform=test_transform)
    if dataset_class == SVHN:
        full_train_dataset = dataset_class(root=root,
                                           split='train',
                                           transform=transform,
                                           download=True)
        test_dataset = dataset_class(root=root,
                                     split='test',
                                     transform=transform,
                                     download=True)
    if dataset_class == ImageNet:
        full_train_dataset = dataset_class(root=f'{root}/ImageNet2',
                                           split='train',
                                           transform=INtrain_transform)
        test_dataset = dataset_class(root=f'{root}/ImageNet2',
                                     split='val',
                                     transform=INtest_transform)

    batch_size = 128 #change to appropriate value for dataset in use

    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    train_loader = DataLoader(full_train_dataset,
                              batch_size=batch_size,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=32) 

    val_loader = DataLoader(full_train_dataset,
                            batch_size=batch_size,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=32) 

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=32) 
    
    return train_loader, val_loader, test_loader, len(test_dataset.classes), transforms.functional.get_image_num_channels(test_dataset[0][0])