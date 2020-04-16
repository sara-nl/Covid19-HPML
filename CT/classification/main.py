import os
import numpy as np
import time
import pdb


from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_train_dataset, get_val_dataset, get_test_dataset, make_weights_for_balanced_classes
from models.predefined_models import load_baseline
import options
from train import train_model, evaluate_model
from utils import print_epoch_progress, get_lr, calc_multi_cls_measures


def main(opts):
    """Main function for the training pipeline
    :opts: commandlien arguments
    :returns: None
    """
    ##########################################################################
    #                             Basic settings                             #
    ##########################################################################
    exp_dir = 'experiments'
    log_dir = os.path.join(exp_dir, 'logs')
    model_dir = os.path.join(exp_dir, 'models')
    os.makedirs(os.path.join(model_dir, opts.run_name), exist_ok=True)

    ##########################################################################
    #  Define all the necessary variables for model training and evaluation  #
    ##########################################################################
    writer = SummaryWriter(os.path.join(log_dir, opts.run_name), flush_secs=5)

    train_dataset = get_train_dataset(root=os.path.join('data', 'train'))
    weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opts.batch_size, num_workers=6,
        drop_last=False, sampler=sampler)

    val_dataset = get_val_dataset(root=os.path.join('data', 'val'))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=6,
        drop_last=False)

    assert train_dataset.class_to_idx == val_dataset.class_to_idx, "Mapping not correct"

    model = load_baseline(n_classes=2)

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=0.1)

    if opts.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
    elif opts.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=opts.gamma)

    best_val_loss = float('inf')
    best_val_accu = float(0)

    iteration_change_loss = 0
    t_start_training = time.time()

    ##########################################################################
    #                           Main training loop                           #
    ##########################################################################
    for epoch in range(opts.epochs):
        current_lr = get_lr(optimizer)
        t_start = time.time()

        ############################################################
        #  The actual training and validation step for each epoch  #
        ############################################################
        train_loss, train_metric = train_model(
            model, train_loader, epoch, opts.epochs, optimizer, writer,
            current_lr, opts.log_every)

        with torch.no_grad():
            val_loss, val_metric = evaluate_model(
                model, val_loader, epoch, opts.epochs, writer, current_lr)

        ##############################
        #  Write to summary writer   #
        ##############################

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_metric['accuracy'], epoch)
        writer.add_scalar('Precision/Train', train_metric['precisions'], epoch)
        writer.add_scalar('Recall/Train', train_metric['recalls'], epoch)
        writer.add_scalar('F1/Train', train_metric['f1'], epoch)

        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_metric['accuracy'], epoch)
        writer.add_scalar('Precision/Val', val_metric['precisions'], epoch)
        writer.add_scalar('Recall/Val', val_metric['recalls'], epoch)
        writer.add_scalar('F1/Val', val_metric['f1'], epoch)

        ##############################
        #  Adjust the learning rate  #
        ##############################
        if opts.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        elif opts.lr_scheduler == 'step':
            scheduler.step()

        t_end = time.time()
        delta = t_end - t_start

        print_epoch_progress(train_loss, val_loss, delta, train_metric,
                             val_metric)
        iteration_change_loss += 1
        print('-' * 30)

        train_acc, val_acc = train_metric['accuracy'], val_metric['accuracy']
        # file_name = ('val_acc_{}_train_acc_{}_epoch_{}.pth'.
        #              format(train_acc, val_acc, epoch))
        # torch.save(model, os.path.join(model_dir, opts.run_name, file_name))

        if val_acc > best_val_accu:
            best_val_accu = val_acc
            if bool(opts.save_model):
                torch.save(model, os.path.join(model_dir, opts.run_name, 'best.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == opts.patience and opts.early_stopping:
            print(('Early stopping after {0} iterations without the decrease ' +
                   'of the val loss').format(iteration_change_loss))
            break
    t_end_training = time.time()
    print('training took {}s'.
          format(t_end_training - t_start_training))


if __name__ == "__main__":
    opts = options.parse_arguments()
    main(opts)
