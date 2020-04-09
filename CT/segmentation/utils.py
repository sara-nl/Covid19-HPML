import numpy as np
import matplotlib.pyplot as plt
import kornia
import torch
import random


def calc_multi_cls_measures(probs, mask):
    """Calculate multi-class classification measures (Accuracy, precision,
    Recall, AUC.
    :probs: NxC numpy array storing probabilities for each case
    :label: ground truth label
    :returns: a dictionary of accuracy, precision and recall
    """
    n_classes = probs.shape[1]
    preds = torch.argmax(probs, axis=1)

    accuracy = (preds.eq(mask).sum() / float(preds.shape[0] * preds.shape[1] * preds.shape[2])).item()
    miou = kornia.utils.metrics.mean_iou(preds, mask, num_classes=n_classes).mean().item()

    metric_collects = {'accuracy': accuracy, 'miou': miou}
    return metric_collects


def write_img(probs, mask, epoch, writer, is_train=False):
    """ Write the predicted and default images"""
    if is_train:
        prep = 'Training '
    else:
        prep = 'Validation '

    idx = random.randint(0, probs.shape[0]-1)
    n_classes = probs.shape[1]

    preds = torch.argmax(probs, axis=1)[idx].numpy() / float(n_classes - 1)
    mask = mask[idx].numpy() / float(n_classes - 1)

    writer.add_image(f'{prep} predicted image', preds, epoch, dataformats='HW')
    writer.add_image(f'{prep} ground truth', mask, epoch, dataformats='HW')


def print_epoch_progress(epoch, n_epoch, train_loss, val_loss, time_duration, train_metric,
                         val_metric):
    """Print all the information after each epoch.
    :train_loss: average training loss
    :val_loss: average validation loss
    :time_duration: time duration for current epoch
    :train_metric_collects: a performance dictionary for training
    :val_metric_collects: a performance dictionary for validation
    :returns: None
    """
    log_str = 'Epoch {}/{}|'.format(epoch, n_epoch)
    train_acc, val_acc = train_metric['accuracy'], val_metric['accuracy']
    train_miou, val_miou = train_metric['miou'], val_metric['miou']
    log_str += 'Train/Val| Loss: {:.4f}/{:.4f}|'.format(train_loss, val_loss)
    log_str += 'Acc: {:.4f}/{:.4f}|'.format(train_acc, val_acc)
    log_str += 'mIoU: {:.4f}/{:.4f}|'.format(train_miou, val_miou)

    log_str += 'T(s) {:.2f}'.format(time_duration)

    print(log_str)
    print('-' * 30)


def reverse_transform(inp):
    """ Do a reverse transformation. inp should be of shape [3, H, W] """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp
