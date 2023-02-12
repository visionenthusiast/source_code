'''
Define classes for CIFAR train and test datasets
load_CIFAR - for loading CIFAR dataset
'''
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TrainSet(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=True, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

class TestSet(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=False, download=True, transform=None):
        super().__init__(root=root, train=False, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

def load_CIFAR(train, batch_size=128):
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    if train == True:
        transform = A.Compose([
            A.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
            A.RandomCrop(32, 32, p=1.0),
            A.ShiftScaleRotate(shift_limit = 0.06, scale_limit = 0.1, rotate_limit = 45),
            A.CoarseDropout(min_holes=1, max_holes=1, min_height=16, max_width=16, min_width=16, max_height=16, fill_value = [127,127,127], mask_fill_value = None, p=1),
            A.Normalize((0.4914, 0.4822, 0.4465),(0.2471, 0.2435, 0.2616)),
            ToTensorV2()
            ])

        dataset = TrainSet(transform=transform)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)
    else:
        transform = A.Compose([
            A.Normalize((0.4914, 0.4822, 0.4465),(0.2471, 0.2435, 0.2616)),
            ToTensorV2()
            ])

        dataset = TestSet(transform=transform)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return classes, data_loader