from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision import transforms
import numpy as np

COLOR = ['c10', 'c100']
BW = ['m']

def get_data(dataset, val_split=0.5):
    if dataset in COLOR:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
    elif dataset.lower() == 'm':
        dataset_class = MNIST
    
    batch_size = 4

    full_train_dataset = dataset_class(root='/data/neelesh/CF',
                                 train=True,
                                 download=True,
                                 transform=transform)

    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    train_loader = DataLoader(full_train_dataset,
                              batch_size=batch_size,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=4) 

    val_loader = DataLoader(full_train_dataset,
                            batch_size=batch_size,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=4) 

    test_dataset = dataset_class(root='/data/neelesh/CF',
                           train=False,
                           download=True,
                           transform=transform)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4) 

    return train_loader, val_loader, test_loader, len(test_dataset.classes), transforms.functional.get_image_num_channels(test_dataset[0][0])

