from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from .caltech import Caltech101
from .config import cfg


def get_loader(dataset, batch_size, num_workers=4, transform_train=None, transform_test=None):
    root = cfg.default_root
    if dataset == 'caltech101':
        Dataset = Caltech101

        if transform_train == None:
            transform_train = transforms.Compose([
                transforms.Resize(128),
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
                # transforms.Normalize((0.5453, 0.5283, 0.5022), (0.2422, 0.2392, 0.2406))
            ])
        if transform_test == None:
            transform_test = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
                # transforms.Normalize((0.5453, 0.5283, 0.5022), (0.2422, 0.2392, 0.2406))
            ])
    else:
        raise NotImplementedError

    data_train = Dataset(root=root,
                          transform=transform_train)
    data_test = Dataset(root=root,
                         train=False,
                         transform=transform_test)

    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    data_test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=num_workers)

    return data_train_loader, data_test_loader
