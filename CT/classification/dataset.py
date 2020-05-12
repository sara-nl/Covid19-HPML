import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def get_train_dataset(root, opts):
    """ Load the COVID classification training dataset as an ImageFolder dataset """
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])

    train_transforms = transforms.Compose([
        transforms.Resize(int(opts.img_size * 1.5)),
        transforms.RandomResizedCrop(opts.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.ToTensor(),
        normalize
    ])

    dataset = torchvision.datasets.ImageFolder(root, transform=train_transforms)
    return dataset


def get_val_dataset(root, opts):
    """ Load the COVID classification validation dataset as an ImageFolder dataset """
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])

    val_transforms = transforms.Compose([
        transforms.Resize((opts.img_size, opts.img_size)),
        transforms.ToTensor(),
        normalize
    ])

    dataset = torchvision.datasets.ImageFolder(root, transform=val_transforms)
    return dataset


def get_test_dataset(root, opts):
    """ Load the COVID classification test dataset as an ImageFolder dataset """
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])

    val_transforms = transforms.Compose([
        transforms.Resize((opts.img_size, opts.img_size)),
        transforms.ToTensor(),
        normalize
    ])

    dataset = torchvision.datasets.ImageFolder(root, transform=val_transforms)
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
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


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


