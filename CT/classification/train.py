import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pdb

import utils


def train_model(model, train_loader, optimizer, opts):
    n_classes = opts.n_classes
    metric = torch.nn.CrossEntropyLoss()

    y_probs = np.zeros((0, n_classes), np.float)
    y_trues = np.zeros((0), np.int)
    losses = []
    model.train()

    for i, (image, label) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        image, label = utils.move_to([image, label], opts.device)

        prediction = model.forward(image.float())
        loss = metric(prediction, label.long())
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        y_prob = F.softmax(prediction, dim=1)

        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

    metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)

    train_loss_epoch = np.round(np.mean(losses), 4)
    return train_loss_epoch, metric_collects


def evaluate_model(model, val_loader, opts):
    n_classes = opts.n_classes
    metric = torch.nn.CrossEntropyLoss()

    model.eval()

    y_probs = np.zeros((0, n_classes), np.float)
    y_trues = np.zeros((0), np.int)
    losses = []

    for i, (image, label) in enumerate(tqdm(val_loader)):
        image, label = utils.move_to([image, label], opts.device)

        prediction = model.forward(image.float())
        loss = metric(prediction, label.long())

        loss_value = loss.item()
        losses.append(loss_value)
        y_prob = F.softmax(prediction, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

    metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)

    val_loss_epoch = np.round(np.mean(losses), 4)
    return val_loss_epoch, metric_collects
