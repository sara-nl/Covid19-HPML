import torchvision
import torch
import os
import copy
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def get_train_dataset_by_oversampling(root, opts, challenge_folder, folder2=None, folder3=None):
    """ Load a COVID classification training dataset in which the data provided by the grand-challenge
        organisers is oversampled. The path to this is given by the 'challenge_folder' argument """

    train_transforms = get_transform(opts, train=True)
    dataset1 = torchvision.datasets.ImageFolder(os.path.join(root, challenge_folder), transform=train_transforms)
    dataset2, dataset3 = None, None

    if folder2 is None and folder3 is None:
        return dataset1

    if folder2 is not None:
        dataset2 = torchvision.datasets.ImageFolder(os.path.join(root, folder2), transform=train_transforms)
        assert dataset1.class_to_idx == dataset2.class_to_idx

    if folder3 is not None:
        dataset3 = torchvision.datasets.ImageFolder(os.path.join(root, folder3), transform=train_transforms)
        assert dataset1.class_to_idx == dataset3.class_to_idx

    datasets = [d for d in [dataset1, dataset2, dataset3] if d is not None]

    dataset_lengths = [len(d) for d in datasets]
    other_dataset_lenghts = sum(dataset_lengths[1:])

    dataset1_multiplier = (other_dataset_lenghts // dataset_lengths[0]) + 1
    dataset1.samples = dataset1.samples * dataset1_multiplier
    dataset1.targets = dataset1.targets * dataset1_multiplier
    dataset1.imgs = dataset1.imgs * dataset1_multiplier

    combined_dataset = torch.utils.data.ConcatDataset([d for d in [dataset1, dataset2, dataset3] if d is not None])
    combined_dataset.class_to_idx = copy.deepcopy(dataset1.class_to_idx)

    return combined_dataset


def get_pretrain_and_finetune_datast(root, opts, finetune_folder, folder2, folder3):
    """ Get the datasets for pretraining and finetuning """

    train_transforms = get_transform(opts, train=True)
    finetune_dataset = torchvision.datasets.ImageFolder(os.path.join(root, finetune_folder), transform=train_transforms)
    dataset2, dataset3 = None, None

    if folder2 is not None:
        dataset2 = torchvision.datasets.ImageFolder(os.path.join(root, folder2), transform=train_transforms)
        assert finetune_dataset.class_to_idx == dataset2.class_to_idx

    if folder3 is not None:
        dataset3 = torchvision.datasets.ImageFolder(os.path.join(root, folder3), transform=train_transforms)
        assert finetune_dataset.class_to_idx == dataset3.class_to_idx

    combined_dataset = torch.utils.data.ConcatDataset([d for d in [dataset2, dataset3] if d is not None])
    combined_dataset.class_to_idx = copy.deepcopy(finetune_dataset.class_to_idx)

    return combined_dataset, finetune_dataset


def get_train_dataset(root, opts, folder1, folder2=None, folder3=None):
    """ Load the COVID classification training dataset as an ImageFolder dataset """

    train_transforms = get_transform(opts, train=True)
    dataset1 = torchvision.datasets.ImageFolder(os.path.join(root, folder1), transform=train_transforms)
    dataset2, dataset3 = None, None

    if folder2 is not None:
        dataset2 = torchvision.datasets.ImageFolder(os.path.join(root, folder2), transform=train_transforms)
        assert dataset1.class_to_idx == dataset2.class_to_idx

    if folder3 is not None:
        dataset3 = torchvision.datasets.ImageFolder(os.path.join(root, folder3), transform=train_transforms)
        assert dataset1.class_to_idx == dataset3.class_to_idx

    combined_dataset = torch.utils.data.ConcatDataset([d for d in [dataset1, dataset2, dataset3] if d is not None])
    combined_dataset.class_to_idx = copy.deepcopy(dataset1.class_to_idx)

    return combined_dataset


def get_val_dataset(root, opts):
    """ Load the COVID classification validation dataset as an ImageFolder dataset """

    val_transforms = get_transform(opts, train=False)
    dataset = torchvision.datasets.ImageFolder(root, transform=val_transforms)

    return dataset


def get_test_dataset(root, opts):
    """ Load the COVID classification test dataset as an ImageFolder dataset """

    test_transforms = get_transform(opts, train=False)
    dataset = torchvision.datasets.ImageFolder(root, transform=test_transforms)

    return dataset


def reverse_transform(inp):
    """ Do a reverse transformation. inp should be of shape [3, H, W] """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.45271412, 0.45271412, 0.45271412])
    std = np.array([0.33165374, 0.33165374, 0.33165374])
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
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def get_transform(opts, train=True):
    """ Function to get the transform for image preprocessing """
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])

    if train:
        transform = transforms.Compose([
            transforms.Resize(int(opts.img_size * 1.5)),
            transforms.RandomResizedCrop(opts.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((opts.img_size, opts.img_size)),
            transforms.ToTensor(),
            normalize
        ])
    return transform


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import skimage
    import matplotlib.pyplot as plt
    import imageio

    train_dataset = get_train_dataset('data/train')
    val_dataset = get_val_dataset('data/val')
    test_dataset = get_test_dataset('data/test')

    train_loader = DataLoader(train_dataset, batch_size=4, drop_last=False, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, drop_last=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, drop_last=False, shuffle=False)

    for batch_index, batch_samples in enumerate(train_loader):
        data, target = batch_samples
        plt.imshow(reverse_transform(data[0]))
        plt.show()
        plt.imshow(data[0].numpy().transpose(1, 2, 0))
        plt.show()
        print()
