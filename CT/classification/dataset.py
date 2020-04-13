import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def get_train_dataset(root):
    """ Load the COVID classification training dataset as an ImageFolder dataset """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    dataset = torchvision.datasets.ImageFolder(root, transform=train_transforms)
    return dataset


def get_val_dataset(root):
    """ Load the COVID classification validation dataset as an ImageFolder dataset """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = torchvision.datasets.ImageFolder(root, transform=val_transforms)
    return dataset


def get_test_dataset(root):
    """ Load the COVID classification test dataset as an ImageFolder dataset """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = torchvision.datasets.ImageFolder(root, transform=test_transforms)
    return dataset


def reverse_transform(inp):
    """ Do a reverse transformation. inp should be of shape [3, H, W] """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


def make_weights_for_balanced_classes(images, num_classes):
    count = [0] * num_classes
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * num_classes
    N = float(sum(count))
    for i in range(num_classes):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


if __name__ == '__main__':
    train_dataset = get_train_dataset('data/train')
    val_dataset = get_val_dataset('data/val')
    test_dataset = get_test_dataset('data/test')
