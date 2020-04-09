import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import kornia
from collections import OrderedDict
import pdb

import utils


def train_model(model, train_loader, epoch, optimizer, writer, opts):
    n_classes = model.n_classes
    metric = nn.CrossEntropyLoss()

    y_probs = torch.zeros(0, n_classes, 512, 512)
    y_trues = torch.zeros(0, 512, 512).long()
    epoch_loss = 0
    model.train()

    for i, (image, mask) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            mask = mask.cuda()

        prediction = model.forward(image)

        # For the torchvision models, an OrderedDict is returned
        if isinstance(prediction, OrderedDict):
            prediction = prediction['out']

        loss = metric(prediction, mask)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        y_prob = F.softmax(prediction.cpu(), dim=1)
        y_probs = torch.cat([y_probs, y_prob.detach().cpu()])
        y_trues = torch.cat([y_trues, mask.cpu()])

    metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)
    utils.write_img(y_probs, y_trues, epoch, writer, is_train=True)

    writer.add_scalar('Training loss', epoch_loss, epoch)
    writer.add_scalar('Training accuracy', metric_collects['accuracy'], epoch)
    writer.add_scalar('Training miou', metric_collects['miou'], epoch)

    return epoch_loss, metric_collects


def evaluate_model(model, val_loader, epoch, writer, opts):
    n_classes = model.n_classes
    metric = torch.nn.CrossEntropyLoss()

    model.eval()

    y_probs = torch.zeros(0, n_classes, 512, 512)
    y_trues = torch.zeros(0, 512, 512).long()
    epoch_loss = 0

    for i, (image, mask) in enumerate(val_loader):

        if torch.cuda.is_available():
            image = image.cuda()
            mask = mask.cuda()

        prediction = model.forward(image)

        # For the torchvision models, an OrderedDict is returned
        if isinstance(prediction, OrderedDict):
            prediction = prediction['out']

        loss = metric(prediction, mask)

        epoch_loss += loss.item()

        y_prob = F.softmax(prediction, dim=1)
        y_probs = torch.cat([y_probs, y_prob.detach().cpu()])
        y_trues = torch.cat([y_trues, mask.cpu()])

    metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)
    utils.write_img(y_probs, y_trues, epoch, writer, is_train=False)

    writer.add_scalar('Validation loss', epoch_loss, epoch)
    writer.add_scalar('Validation accuracy', metric_collects['accuracy'], epoch)
    writer.add_scalar('Validation miou', metric_collects['miou'], epoch)

    return epoch_loss, metric_collects
