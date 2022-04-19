from torch.utils.data import DataLoader
from torchvision import datasets, transforms


dataset_loadpath = {
    'MNIST': '/datasets/mnist',
    'MSTAR_train': '/datasets/cifar10/Mstar/train',
    'MSTAR_val': '/datasets/cifar10/Mstar/val',
    'CIFAR10': '/datasets/cifar10',
    'ImageNet_train': '/datasets/imagenet/train',
    'ImageNet_val': '/datasets/imagenet/val'
}


def get_MNIST_dataset(batch_size=256, devices=1):
    '''
    Get the dataloader of the MNIST dataset
    :param batch_size: batch size
    :param devices: number of devices (GPU)
    :return: MNIST train and test dataloader
    '''
    train_dataset = datasets.MNIST(
        root=dataset_loadpath['MNIST'],
        train=True,
        transform=transforms.Compose([
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        download=False
    )
    test_dataset = datasets.MNIST(
        root=dataset_loadpath['MNIST'],
        train=False,
        transform=transforms.Compose([
            ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        download=False
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size * devices,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size * devices,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None
    )
    return train_dataloader, test_dataloader


def get_MSTAR_dataset(batch_size=128, devices=1, resize=False):
    '''
    Get the dataloader of the MSTAR dataset
    :param batch_size: batch_size
    :param devices: number of devices (GPU)
    :return: MSTAR train and val dataloader
    '''
    if resize:
        train_dataset = datasets.ImageFolder(
            root=dataset_loadpath['MSTAR_train'],
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        )
        val_dataset = datasets.ImageFolder(
            root=dataset_loadpath['MSTAR_val'],
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        )
    else:
        train_dataset = datasets.ImageFolder(
            root='/datasets/cifar10/Mstar/train',
            transform=transforms.ToTensor()
        )
        val_dataset = datasets.ImageFolder(
            root='/datasets/cifar10/Mstar/val',
            transform=transforms.ToTensor()
        )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size * devices,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size * devices,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None
    )
    return train_dataloader, val_dataloader


def get_CIFAR10_dataset(batch_size=128, devices=1):
    '''
    Get the dataloader of the CIFAR10 dataset
    :param batch_size: batch size
    :param devices: number of devices (GPU)
    :return: CIFAR10 train and test dataloader
    '''
    train_dataset = datasets.CIFAR10(
        root=dataset_loadpath['CIFAR10'],
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    test_dataset = datasets.CIFAR10(
        root=dataset_loadpath['CIFAR10'],
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size * devices,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size * devices,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None
    )
    return train_dataloader, test_dataloader


def get_CIFAR100_dataset(batch_size=128, devices=1):
    '''
    Get the dataloader of the CIFAR100 dataset
    :param batch_size: batch size
    :param devices: number of devices (GPU)
    :return: CIFAR100 train and test dataloader
    '''
    train_dataset = datasets.CIFAR100(
        root='/rdata/datasets/cifar100/',
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    test_dataset = datasets.CIFAR100(
        root='/rdata/datasets/cifar100',
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size * devices,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size * devices,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None
    )
    return train_dataloader, test_dataloader


def get_ImageNet_dataset(batch_size=128, devices=1):
    '''
    Get the dataloader of the ImageNet dataset
    :param batch_size: batch size
    :param devices: number of devices (GPU)
    :return: ImageNet train and val dataloader
    '''
    train_dataset = datasets.ImageFolder(
        root=dataset_loadpath['ImageNet_train'],
        transform=transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )
    val_dataset = datasets.ImageFolder(
        root=dataset_loadpath['ImageNet_val'],
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size * devices,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        sampler=None
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size * devices,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        sampler=None
    )
    return train_dataloader, val_dataloader