import os
import numpy as np
import time
import json
import random
from pprint import pprint
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
from collections import defaultdict
import pdb

from dataset import get_train_dataset, get_train_dataset_by_oversampling, get_pretrain_and_finetune_datast, \
    get_val_dataset, get_test_dataset
from models.model_utils import get_model
import options
from train import train_model, evaluate_model
from utils import print_epoch_progress, get_lr, get_lr_scheduler


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
    os.makedirs(os.path.join(log_dir, opts.run_name))

    pprint(vars(opts))
    with open(os.path.join(log_dir, opts.run_name, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    ##########################################################################
    #  Define all the necessary variables for model training and evaluation  #
    ##########################################################################
    writer = SummaryWriter(os.path.join(log_dir, opts.run_name), flush_secs=5)

    if opts.train_mode == 'combined':
        train_dataset = get_train_dataset(opts.data_root, opts, opts.folder1, opts.folder2, opts.folder3)
    elif opts.train_mode == 'oversampling':
        train_dataset = get_train_dataset_by_oversampling(opts.data_root, opts, opts.folder1, opts.folder2,
                                                          opts.folder3)
    elif opts.train_mode == 'pretrain_and_finetune':
        train_dataset, finetune_dataset = get_pretrain_and_finetune_datast(opts.data_root, opts, opts.folder1,
                                                                           opts.folder2, opts.folder3)
        finetune_loader = torch.utils.data.DataLoader(
            finetune_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, drop_last=False, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, drop_last=False, shuffle=True)

    val_dataset = get_val_dataset(os.path.join('data', 'val'), opts)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opts.eval_batch_size, shuffle=False, num_workers=opts.num_workers, drop_last=False)

    test_dataset = get_test_dataset(os.path.join('data', 'test'), opts)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opts.eval_batch_size, shuffle=False, num_workers=opts.num_workers, drop_last=False)

    assert train_dataset.class_to_idx == val_dataset.class_to_idx == test_dataset.class_to_idx, "Mapping not correct"

    model = get_model(opts)

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1 and not opts.no_data_parallel:
        model = nn.DataParallel(model)

    model = model.to(opts.device)

    optimizer = optim.RMSprop(model.parameters(), lr=opts.lr, alpha=0.9, weight_decay=1e-5, momentum=0.9)
    scheduler = get_lr_scheduler(optimizer, opts)

    best_val_loss = float('inf')
    best_val_accu = float(0)
    best_val_rec = float(0)
    best_val_prec = float(0)
    best_val_f1 = float(0)
    best_val_auc = float(0)

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
        train_loss, train_metric = train_model(model, train_loader, optimizer, opts)

        if epoch == opts.finetune_epoch and opts.train_mode == 'pretrain_and_finetune':
            train_loader = finetune_loader
            optimizer = optim.RMSprop(model.parameters(), lr=opts.lr, alpha=0.9, weight_decay=1e-5, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size_finetuning, gamma=opts.gamma)

        # Run the validation set
        with torch.no_grad():
            val_loss, val_metric = evaluate_model(model, val_loader, opts)

        ##############################
        #  Write to summary writer   #
        ##############################

        train_acc, val_acc = train_metric['accuracy'], val_metric['accuracy']
        train_rec, val_rec = train_metric['recalls'], val_metric['recalls']
        train_prec, val_prec = train_metric['precisions'], val_metric['precisions']
        train_f1, val_f1 = train_metric['f1'], val_metric['f1']
        train_auc, val_auc = train_metric['auc'], val_metric['auc']

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Precision/Train', train_prec, epoch)
        writer.add_scalar('Recall/Train', train_rec, epoch)
        writer.add_scalar('F1/Train', train_f1, epoch)
        writer.add_scalar('AUC/Train', train_auc, epoch)

        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Precision/Val', val_prec, epoch)
        writer.add_scalar('Recall/Val', val_rec, epoch)
        writer.add_scalar('F1/Val', val_f1, epoch)
        writer.add_scalar('AUC/Val', val_auc, epoch)

        ##############################
        #  Adjust the learning rate  #
        ##############################
        if opts.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        elif opts.lr_scheduler in ['step', 'cosine']:
            scheduler.step()

        t_end = time.time()
        delta = t_end - t_start

        print_epoch_progress(epoch, opts.epochs, train_loss, val_loss,
                             delta, train_metric, val_metric)
        iteration_change_loss += 1
        print('-' * 30)

        if val_acc > best_val_accu:
            best_val_accu = val_acc
            if bool(opts.save_model):
                torch.save(model.state_dict(), os.path.join(model_dir, opts.run_name, 'best_state_dict.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if val_rec > best_val_rec:
            best_val_rec = val_rec

        if val_prec > best_val_prec:
            best_val_prec = val_prec

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f'The best validation F1-score is now {best_val_f1}')
            print(f'The validation accuracy and AUC are now {val_acc} and {val_auc}')

        if val_auc > best_val_auc:
            best_val_auc = val_auc

        if iteration_change_loss == opts.patience and opts.early_stopping:
            print(('Early stopping after {0} iterations without the decrease ' +
                   'of the val loss').format(iteration_change_loss))
            break

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training}s')
    print(f'Best validation accuracy: {best_val_accu}')
    print(f'Best validation loss: {best_val_loss}')
    print(f'Best validation precision: {best_val_prec}')
    print(f'Best validation recall: {best_val_rec}')
    print(f'Best validation f1: {best_val_f1}')
    print(f'Best validation AUC: {best_val_auc}')

    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(model_dir, opts.run_name, 'best_state_dict.pth')))
        test_loss, test_metric = evaluate_model(model, test_loader, opts)

    print(f'The best test F1: {test_metric["f1"]}')
    print(f'The best test auc: {test_metric["auc"]}')
    print(f'The best test accuracy: {test_metric["accuracy"]}')


if __name__ == "__main__":
    opts = options.parse_arguments()
    main(opts)
