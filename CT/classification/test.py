import os
import numpy as np
import time
import pdb
import torch
import torch.nn.functional as F
import argparse

import utils
from dataset import get_train_dataset, get_val_dataset, get_test_dataset, make_weights_for_balanced_classes
from models.model_utils import get_model


def test_model(model, test_loader):

    n_classes = model.n_classes
    metric = torch.nn.CrossEntropyLoss()

    model.eval()

    y_probs = np.zeros((0, n_classes), np.float)
    y_trues = np.zeros((0), np.int)
    losses = []

    for i, (image, label) in enumerate(test_loader):

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        prediction = model.forward(image.float())
        loss = metric(prediction, label.long())

        loss_value = loss.item()
        losses.append(loss_value)
        y_prob = F.softmax(prediction, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

    metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)

    test_loss_epoch = np.round(np.mean(losses), 4)
    return test_loss_epoch, metric_collects


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['dense', 'covidnet_small', 'covidnet_large'],
                        default='covidnet_large')
    parser.add_argument('--path_to_model', type=str, default='experiments/models/large_512_20200421T134932/best.pth')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    model = get_model(args)

    test_dataset = get_test_dataset(os.path.join('data', 'test'), args)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count(),
        drop_last=False)

    with torch.no_grad():
        test_loss_epoch, test_metric = test_model(model, test_loader)
        
    print(f'The best test F1: {test_metric["f1"]}')
    print(f'The best test auc: {test_metric["auc"]}')
    print(f'The best test accuracy: {test_metric["accuracy"]}')
    print(f'The best test recall: {test_metric["recalls"]}')
    print(f'The best test precision: {test_metric["precisions"]}')
