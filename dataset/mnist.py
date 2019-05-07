import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def MNIST(data_config):
    if 'data_folder' in data_config:
        data_folder = data_config['data_folder']
    else:
        data_folder = './data/'
    train_transform = [transforms.ToTensor()]
    test_transform = [transforms.ToTensor()]
    if 'normalize' in data_config and data_config['normalize']:
        train_transform.append(transforms.Normalize((0.1307,), (0.3081,)))
        test_transform.append(transforms.Normalize((0.1307,), (0.3081,)))
    if 'flip' in data_config and data_config['flip']:
        train_transform.append(transforms.RandomHorizontalFlip())
    if 'crop' in data_config and data_config['crop']:
        train_transform.insert(0,transforms.RandomCrop(28,padding=2,padding_mode='edge'))
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    N = 50000
    N_valid = data_config['validation_size']
    indexes = np.random.permutation(N)
    valid_idx = indexes[:N_valid]
    train_idx = indexes[N_valid:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(datasets.MNIST(data_folder, train=True, download=True,transform=train_transform),
                        batch_size=data_config['batch_size'], sampler=train_sampler, num_workers=data_config['num_workers'])

    valid_loader = torch.utils.data.DataLoader(datasets.MNIST(data_folder, train=True, download=True, transform=test_transform),
                        batch_size=data_config['batch_size'], sampler=valid_sampler, num_workers=data_config['num_workers'])

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(data_folder, train=False, download=True, transform=test_transform),
                        batch_size=data_config['batch_size'], shuffle=False, num_workers=data_config['num_workers'])

    return train_loader, valid_loader, test_loader



