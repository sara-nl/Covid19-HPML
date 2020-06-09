import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pdb


def calc_multi_cls_measures(probs, label):
    """Calculate multi-class classification measures (Accuracy, precision,
    Recall, AUC.
    :probs: NxC numpy array storing probabilities for each case
    :label: ground truth label
    :returns: a dictionary of accuracy, precision and recall
    """
    label = label.reshape(-1, 1)
    n_classes = probs.shape[1]
    preds = np.argmax(probs, axis=1)
    accuracy = accuracy_score(label, preds)
    precisions = precision_score(label, preds, average='binary', pos_label=1,
                                 labels=range(n_classes), zero_division=0.)
    recalls = recall_score(label, preds, average='binary', labels=range(n_classes),
                           zero_division=0., pos_label=1)
    f1s = f1_score(label, preds, average='binary', pos_label=1, labels=range(n_classes),
                   zero_division=0.)
    aucs = roc_auc_score(label, preds)

    metric_collects = {'accuracy': accuracy, 'precisions': precisions,
                       'recalls': recalls, 'f1': f1s, 'auc': aucs}
    return metric_collects


def print_progress(epoch=None, n_epoch=None, n_iter=None, iters_one_batch=None,
                   mean_loss=None, cur_lr=None, metric_collects=None,
                   prefix=None):
    """Print the training progress.
    :epoch: epoch number
    :n_epoch: total number of epochs
    :n_iter: current iteration number
    :mean_loss: mean loss of current batch
    :iters_one_batch: number of iterations per batch
    :cur_lr: current learning rate
    :metric_collects: dictionary returned by function calc_multi_cls_measures
    :returns: None
    """
    accuracy = metric_collects['accuracy']
    precisions = metric_collects['precisions']
    recalls = metric_collects['recalls']

    log_str = ''
    if epoch is not None:
        log_str += 'Ep: {0}/{1}|'.format(epoch, n_epoch)

    if n_iter is not None:
        log_str += 'It: {0}/{1}|'.format(n_iter, iters_one_batch)

    if mean_loss is not None:
        log_str += 'Loss: {0:.4f}|'.format(mean_loss)

    log_str += 'Acc: {:.4f}|'.format(accuracy)
    templ = 'Pr: ' + ', '.join(['{:.4f}'] * 2) + '|'
    log_str += templ.format(*(precisions[1:].tolist()))
    templ = 'Re: ' + ', '.join(['{:.4f}'] * 2) + '|'
    log_str += templ.format(*(recalls[1:].tolist()))

    if cur_lr is not None:
        log_str += 'lr: {0}'.format(cur_lr)
    log_str = log_str if prefix is None else prefix + log_str
    print(log_str)


def print_epoch_progress(epoch, n_epochs, train_loss, val_loss,
                         time_duration, train_metric, val_metric):
    """Print all the information after each epoch.
    :epoch: int, the currents epoch
    :n_epochs: int, the total number of epochs
    :train_loss: average training loss
    :val_loss: average validation loss
    :time_duration: time duration for current epoch
    :train_metric_collects: a performance dictionary for training
    :val_metric_collects: a performance dictionary for validation
    :returns: None
    """
    log_str = 'Epoch {}/{}|'.format(epoch, n_epochs)

    train_acc, val_acc = train_metric['accuracy'], val_metric['accuracy']
    train_prec, val_prec = train_metric['precisions'], val_metric['precisions']
    train_recalls, val_recalls = train_metric['recalls'], val_metric['recalls']
    train_f1, val_f1 = train_metric['f1'], val_metric['f1']
    train_auc, val_auc = train_metric['auc'], val_metric['auc']

    log_str += 'Train/Val| Loss: {:.4f}/{:.4f}|'.format(train_loss, val_loss)
    log_str += 'Acc: {:.4f}/{:.4f}|'.format(train_acc, val_acc)

    templ = 'Pr: ' + ', '.join(['{:.4f}'] * 1) + '/'
    log_str += templ.format(train_prec)
    templ = ', '.join(['{:.4f}'] * 1) + '|'
    log_str += templ.format(val_prec)

    templ = 'Re: ' + ', '.join(['{:.4f}'] * 1) + '/'
    log_str += templ.format(train_recalls)
    templ = ', '.join(['{:.4f}'] * 1) + '|'
    log_str += templ.format(val_recalls)

    templ = 'F1: ' + ', '.join(['{:.4f}'] * 1) + '/'
    log_str += templ.format(train_f1)
    templ = ', '.join(['{:.4f}'] * 1) + '|'
    log_str += templ.format(val_f1)

    templ = 'AUC: ' + ', '.join(['{:.4f}'] * 1) + '/'
    log_str += templ.format(train_auc)
    templ = ', '.join(['{:.4f}'] * 1) + '|'
    log_str += templ.format(val_auc)

    log_str += 'T(s) {:.2f}'.format(time_duration)
    print(log_str)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    elif isinstance(var, list):
        return [move_to(v, device) for v in var]
    elif isinstance(var, tuple):
        return (move_to(v, device) for v in var)
    return var.to(device)
