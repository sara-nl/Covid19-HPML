import os
import numpy as np
import time
from pprint import pprint
import json

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import load_model
from dataset import SegmentationDataset
from train import train_model, evaluate_model
from options import parse_arguments
import utils


def main(opts):
    """Main function for the training pipeline
    :opts: commandline arguments
    :returns: None
    """
    pprint(vars(opts))
    ##########################################################################
    #                             Basic settings                             #
    ##########################################################################
    exp_dir = 'experiments'
    log_dir = os.path.join(exp_dir, 'logs')
    model_dir = os.path.join(exp_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    os.makedirs(os.path.join(log_dir, opts.run_name), exist_ok=True)
    with open(os.path.join(log_dir, opts.run_name, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    ##########################################################################
    #  Define all the necessary variables for model training and evaluation  #
    ##########################################################################
    writer = SummaryWriter(os.path.join(log_dir, opts.run_name))

    train_dataset = SegmentationDataset(is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opts.batch_size,
                                               num_workers=4,
                                               drop_last=False,
                                               shuffle=True)

    val_dataset = SegmentationDataset(is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=20,
                                             shuffle=False,
                                             num_workers=0,
                                             drop_last=False)

    model = load_model(opts, n_classes=4)
    
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=0.1)

    if opts.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=opts.patience, factor=.3, threshold=0.1, verbose=True)
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
        t_start = time.time()

        ############################################################
        #  The actual training and validation step for each epoch  #
        ############################################################
        train_loss, train_metric = train_model(model, train_loader, epoch, optimizer, writer, opts)

        with torch.no_grad():
            val_loss, val_metric = evaluate_model(model, val_loader, epoch, writer, opts)

            ##############################
            #  Adjust the learning rate  #
            ##############################
            if opts.lr_scheduler == 'plateau':
                scheduler.step(val_loss)
            elif opts.lr_scheduler == 'step':
                scheduler.step()

        t_end = time.time()
        delta = t_end - t_start

        utils.print_epoch_progress(epoch, opts.epochs, train_loss, val_loss, delta, train_metric,
                                   val_metric)

    t_end_training = time.time()
    print('training took {}s'.
          format(t_end_training - t_start_training))


if __name__ == '__main__':
    opts = parse_arguments()
    main(opts)
