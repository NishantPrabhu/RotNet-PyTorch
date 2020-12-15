
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import torch


DATASETS = {
    'cifar10': {'data': datasets.CIFAR10, 'classes': 10},
    'cfiar100': {'data': datasets.CIFAR100, 'classes': 100},
}


def sample_weights(labels):
    """
    Computes sample weights for sampler in dataloaders,
    based on class occurence.
    """
    cls_count = np.unique(labels, return_counts=True)[1]
    cls_weights = 1./torch.Tensor(cls_count)
    return cls_weights[list(map(int, labels))]


class RotnetDataset(torch.utils.data.Dataset):

    def __init__(self, config, split):
        name = config.get('name', None)
        root = config.get('root', './')
        assert name in DATASETS.keys(), f'invalid dataset name {name}'

        if split == 'train':
            self.data = DATASETS[name]['data'](root=root, train=True, transform=None, download=True)
        elif split == 'val':
            self.data = DATASETS[name]['data'](root=root, train=False, transform=None, download=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config['mean'], config['std'])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, target = self.data[i]
        rot0 = self.transform(img).unsqueeze(0)
        rot90 = self.transform(img.rotate(90)).unsqueeze(0)
        rot180 = self.transform(img.rotate(180)).unsqueeze(0)
        rot270 = self.transform(img.rotate(270)).unsqueeze(0)
        images = torch.cat((rot0, rot90, rot180, rot270), dim=0)
        labels = torch.LongTensor([0, 1, 2, 3])
        return {'img': images, 'target': labels}


def get_dataset(config, split):
    name = config.get('name', None)
    root = config.get('root', './')
    assert name in DATASETS.keys(), f'invalid dataset name {name}'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std'])
    ])
    
    if split == 'train':
        data = DATASETS[name]['data'](root=root, train=True, transform=transform, download=True)
    elif split == 'val':
        data = DATASETS[name]['data'](root=root, train=False, transform=transform, download=True)
   
    return data
            

def get_dataloader(dataset, batch_size, num_workers=1, collate_fn=None, shuffle=False, weigh=False, drop_last=False):
    """ Returns a DataLoader with specified configuration """

    if weigh:
        weights = sample_weights(dataset.targets)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, drop_last=drop_last, collate_fn=collate_fn)
    else:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn)
