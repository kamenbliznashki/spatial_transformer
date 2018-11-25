import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T


def fetch_dataloader(params, train=True, mini_size=128):

    # load dataset and init in the dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = MNIST(root=params.data_dir, train=train, download=False, transform=transforms)

    if params.dict.get('mini_data'):
        if train:
            dataset.train_data = dataset.train_data[:mini_size]
            dataset.train_labels = dataset.train_labels[:mini_size]
        else:
            dataset.test_data = dataset.test_data[:mini_size]
            dataset.test_labels = dataset.test_labels[:mini_size]

    if params.dict.get('mini_ones'):
        if train:
            labels = dataset.train_labels[:2000]
            mask = labels==1
            dataset.train_labels = labels[mask][:mini_size]
            dataset.train_data = dataset.train_data[:2000][mask][:mini_size]
        else:
            labels = dataset.test_labels[:2000]
            mask = labels==1
            dataset.test_labels = labels[mask][:mini_size]
            dataset.test_data = dataset.test_data[:2000][mask][:mini_size]

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() and params.device.type is 'cuda' else {}

    return DataLoader(dataset, batch_size=params.batch_size, shuffle=True, drop_last=True, **kwargs)


