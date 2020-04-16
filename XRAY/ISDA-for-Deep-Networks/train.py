import argparse
import os
import shutil
import time
import errno
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import transforms
import torchvision.datasets as datasets
from autoaugment import CIFAR10Policy
from cutout import Cutout
from ISDA import EstimatorCV, ISDALoss

import networks.resnet
import networks.wideresnet
import networks.se_resnet
import networks.se_wideresnet
import networks.densenet_bc
import networks.shake_pyramidnet
import networks.resnext
import networks.shake_shake
import numpy as np
from tpot import TPOTClassifier
import glob
import os
import shutil
# from shutil import copyfile
from pprint import pprint
import pickle
from PIL import Image
import numpy as np
from autoPyTorch import AutoNetClassification, HyperparameterSearchSpaceUpdates
import sklearn.model_selection
import sklearn.metrics
from sklearn import preprocessing
import torch
from torch.autograd import Variable
from torchvision import transforms
import json
from sklearn.externals.joblib import Parallel, delayed
from dask.distributed import Client
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform

input_size = (32, 32)
channels = 3
balance = 280 #Examples / class, set None if you want to load all examples
final_dataset_location = "/tmp/covid_dataset_damian"
initial_dataset_location = "../data"
classes = ["covid", "Pneumonia", "No Finding"]
preset = "full_cs"
save_output_to = "/tmp/{2}class_{1}balanced_ba_ce_{0}_{3}".format(input_size[0], balance, len(classes), preset)

le = None #global le

def get_references(images = [] , labels = [] , filter = "_positive.txt", copy_to = final_dataset_location, copy_from = initial_dataset_location, balance = balance, classes = classes, img_channels = channels):
    '''
    If you don't  want to copy set copy_to = None; only if this is set img_channels is taken into account 
    balance = 180 means each class gets 180 examples before splits
    '''
    for name in glob.glob(initial_dataset_location + os.path.sep + "*{}".format(filter)):
        class_name = name.split(filter)[0].split("/")[-1].lower()
        if any(class_name in s.lower() for s in classes):
            for i, imagepaths in enumerate(open(name).readlines()):
                if i==balance:
                    break
                images.append(imagepaths)
                labels.append(class_name)
    # le = preprocessing.LabelBinarizer()
    global le
    le = preprocessing.LabelEncoder()
    le.fit(list(set(labels)))
    #Create sklearn type labels
    labels = le.transform(labels)
    print("Found {} examples with labels {}".format(len(labels), le.classes_))
    assert len(labels) > 0, "No data found"
    #Copy data locally and preprocess
    if copy_to:
        try:
            shutil.rmtree(copy_to)
        except FileNotFoundError as e:
            pass
        os.makedirs(copy_to, exist_ok=True)
        for i, im in enumerate(images):
            if "\n" in im:
                im = im.strip()
            if img_channels == 3:
                mode = "RGB"
            elif img_channels == 1:
                mode = "L"
            with Image.open(im).convert(mode) as image:
                image = image.resize(input_size)
                im_arr = np.fromstring(image.tobytes(), dtype=np.uint8) #/ 255.0
            try:
                im_arr = im_arr.reshape((image.size[1], image.size[0], img_channels))
            except ValueError as e:
                im_arr = im_arr.reshape((image.size[1], image.size[0]))
                im_arr = np.stack((im_arr,) * img_channels, axis=-1)
            finally:
                im_arr = np.moveaxis(im_arr, -1, 0) #Pytorch is channelfirst
                dest = os.path.join(copy_to, "{0}_{2}.{1}".format(str(i), im.split("/")[-1].split(".")[-1], "-".join(le.inverse_transform(labels)[i].split(" "))))
                # copyfile(im, dest)
                image.save(dest) 
                # print("Wrote image {} of shape {} with label {}({})".format(dest, im_arr.shape, labels[i], le.inverse_transform(labels)[i]))
            images[i] = dest
        print("Copy {} files".format(i+1))
    assert len(images) == len(labels)
    return images, labels, le


class CovidDataset(Dataset):

    def __init__(self, transform=None):
        """
        Args:
        """
        self.references = get_references()
        self.transform = transform

    def __len__(self):
        return len(self.references[0])

    def __getitem__(self, idx):
        image = io.imread(self.references[0][idx]).astype(np.float32)
        image = np.moveaxis(image, -1, 0) #Pytorch is channelfirst
        image = torch.tensor(image)
        label = self.references[1][idx]
        label = torch.tensor(label)
        return image 
        # return (image, label)


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')

parser.add_argument('--model', default='', type=str,
                    help='deep networks to be trained')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')

parser.add_argument('--layers', default=0, type=int,
                    help='total number of layers (have to be explicitly given!)')

parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.set_defaults(augment=True)

parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--name', default='', type=str,
                    help='name of experiment')
parser.add_argument('--no', default='1', type=str,
                    help='index of the experiment (for recording convenience)')

parser.add_argument('--combine-ratio', default=0.5, type=float,
                    help='hyper-patameter_\lambda for ISDA')

# Wide-ResNet & Shake Shake
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor for wideresnet (default: 10)')

# ResNeXt
parser.add_argument('--cardinality', default=8, type=int,
                    help='cardinality for resnext (default: 8)')

# DenseNet
parser.add_argument('--growth-rate', default=12, type=int,
                    help='growth rate for densenet_bc (default: 12)')
parser.add_argument('--compression-rate', default=0.5, type=float,
                    help='compression rate for densenet_bc (default: 0.5)')
parser.add_argument('--bn-size', default=4, type=int,
                    help='cmultiplicative factor of bottle neck layers for densenet_bc (default: 4)')

# Shake_PyramidNet
parser.add_argument('--alpha', default=200, type=int,
                    help='hyper-parameter alpha for shake_pyramidnet')

# Autoaugment
parser.add_argument('--autoaugment', dest='autoaugment', action='store_true',
                    help='whether to use autoaugment')
parser.set_defaults(autoaugment=False)

# cutout
parser.add_argument('--cutout', dest='cutout', action='store_true',
                    help='whether to use cutout')
parser.set_defaults(cutout=False)
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')

# Cosine learning rate
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true',
                    help='whether to use cosine learning rate')
parser.set_defaults(cos_lr=False)

args = parser.parse_args()


# Configurations adopted for training deep networks.
# (specialized for each type of models)
training_configurations = {
    'resnet': {
        'epochs': 160,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'wideresnet': {
        'epochs': 200,
        'batch_size': 16,
        'initial_learning_rate': 0.1,
        'changing_lr': [60, 120, 160, 200],
        'lr_decay_rate': 0.2,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 5e-4,
    },
    'se_resnet': {
        'epochs': 200,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'changing_lr': [80, 120, 160],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'se_wideresnet': {
        'epochs': 240,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'changing_lr': [60, 120, 160, 200],
        'lr_decay_rate': 0.2,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 5e-4,
    },
    'densenet_bc': {
        'epochs': 300,
        'batch_size': 16,
        'initial_learning_rate': 0.1,
        'changing_lr': [150, 200, 250],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'shake_pyramidnet': {
        'epochs': 1800,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'changing_lr': [],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'resnext': {
        'epochs': 350,
        'batch_size': 128,
        'initial_learning_rate': 0.05,
        'changing_lr': [150, 225, 300],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 5e-4,
    },
    'shake_shake': {
        'epochs': 1800,
        'batch_size': 64,
        'initial_learning_rate': 0.1,
        'changing_lr': [],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'shake_shake_x': {
        'epochs': 1800,
        'batch_size': 64,
        'initial_learning_rate': 0.1,
        'changing_lr': [],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
}



record_path = './ISDA test/' + str(args.dataset) \
              + '_' + str(args.model) \
              + '-' + str(args.layers) \
              + (('-' + str(args.widen_factor)) if 'wide' in args.model else '') \
              + (('-' + str(args.widen_factor)) if 'shake_shake' in args.model else '') \
              + (('-' + str(args.growth_rate)) if 'dense' in args.model else '') \
              + (('-' + str(args.alpha)) if 'pyramidnet' in args.model else '') \
              + (('-' + str(args.cardinality)) if 'resnext' in args.model else '') \
              + '_' + str(args.name) \
              + '/' + 'no_' + str(args.no) \
              + '_combine-ratio_' + str(args.combine_ratio) \
              + ('_standard-Aug_' if args.augment else '') \
              + ('_dropout_' if args.droprate > 0 else '') \
              + ('_autoaugment_' if args.autoaugment else '') \
              + ('_cutout_' if args.cutout else '') \
              + ('_cos-lr_' if args.cos_lr else '')

record_file = record_path + '/training_process.txt'
accuracy_file = record_path + '/accuracy_epoch.txt'
loss_file = record_path + '/loss_epoch.txt'
check_point = os.path.join(record_path, args.checkpoint)

def main():

    global best_prec1
    best_prec1 = 0

    global val_acc
    val_acc = []

    global class_num

    class_num = len(classes)

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        if args.autoaugment:
            print('Autoaugment')
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(input_size[0]),
                transforms.RandomHorizontalFlip(), CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(n_holes=args.n_holes, length=args.length),
                normalize,
            ])

        elif args.cutout:
            print('Cutout')
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(input_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Cutout(n_holes=args.n_holes, length=args.length),
                normalize,
            ])

        else:
            print('Standrad Augmentation!')
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(input_size[0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    # assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    coviddata = CovidDataset()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(coviddata, coviddata.references[1], test_size=0.1, random_state=9, shuffle=True)
    train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)),
        batch_size=training_configurations[args.model]['batch_size'], shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)),
        batch_size=len(y_test), shuffle=True, **kwargs)

    # create model
    if args.model == 'resnet':
        model = eval('networks.resnet.resnet' + str(args.layers) + '_cifar')(dropout_rate=args.droprate)
    elif args.model == 'se_resnet':
        model = eval('networks.se_resnet.resnet' + str(args.layers) + '_cifar')(dropout_rate=args.droprate)
    elif args.model == 'wideresnet':
        model = networks.wideresnet.WideResNet(args.layers, len(classes),
                            args.widen_factor, dropRate=args.droprate)
    elif args.model == 'se_wideresnet':
        model = networks.se_wideresnet.WideResNet(args.layers, len(classes),
                            args.widen_factor, dropRate=args.droprate)

    elif args.model == 'densenet_bc':
        model = networks.densenet_bc.DenseNet(growth_rate=args.growth_rate,
                                              block_config=(int((args.layers - 4) / 6),) * 3,
                                              compression=args.compression_rate,
                                              num_init_features=24,
                                              bn_size=args.bn_size,
                                              drop_rate=args.droprate,
                                              small_inputs=True,
                                              efficient=False)
    elif args.model == 'shake_pyramidnet':
        model = networks.shake_pyramidnet.PyramidNet(dataset=args.dataset, depth=args.layers, alpha=args.alpha, num_classes=class_num, bottleneck = True)

    elif args.model == 'resnext':
        if args.cardinality == 8:
            model = networks.resnext.resnext29_8_64(class_num)
        if args.cardinality == 16:
            model = networks.resnext.resnext29_16_64(class_num)

    elif args.model == 'shake_shake':
        if args.widen_factor == 112:
            model = networks.shake_shake.shake_resnet26_2x112d(class_num)
        if args.widen_factor == 32:
            model = networks.shake_shake.shake_resnet26_2x32d(class_num)
        if args.widen_factor == 96:
            model = networks.shake_shake.shake_resnet26_2x32d(class_num)

    elif args.model == 'shake_shake_x':

        model = networks.shake_shake.shake_resnext29_2x4x64d(class_num)

    if not os.path.isdir(check_point):
        mkdir_p(check_point)

    fc = Full_layer(int(model.feature_num), class_num)

    print('Number of final features: {}'.format(
        int(model.feature_num))
    )

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
        + sum([p.data.nelement() for p in fc.parameters()])
    ))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    isda_criterion = ISDALoss(int(model.feature_num), class_num).cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                {'params': fc.parameters()}],
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay'])

    model = torch.nn.DataParallel(model).cuda()
    fc = nn.DataParallel(fc).cuda()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        fc.load_state_dict(checkpoint['fc'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        isda_criterion = checkpoint['isda_criterion']
        val_acc = checkpoint['val_acc']
        best_prec1 = checkpoint['best_acc']
        np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0

    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):

        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train(train_loader, model, fc, isda_criterion, optimizer, epoch)

        # evaluate on validation set

        #last epoch also get confusion matrix and score
        if epoch == training_configurations[args.model]['epochs'] - 1:
            final_epoch = True
        else:
            final_epoch = False
        prec1 = validate(val_loader, model, fc, ce_criterion, epoch, final_epoch)


        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'fc': fc.state_dict(),
            'best_acc': best_prec1,
            'optimizer': optimizer.state_dict(),
            'isda_criterion': isda_criterion,
            'val_acc': val_acc,

        }, is_best, checkpoint=check_point)
        print('Best accuracy: ', best_prec1)
        np.savetxt(accuracy_file, np.array(val_acc))


    print('Best accuracy: ', best_prec1)
    print('Average accuracy', sum(val_acc[len(val_acc) - 10:]) / 10)
    # val_acc.append(sum(val_acc[len(val_acc) - 10:]) / 10)
    # np.savetxt(val_acc, np.array(val_acc))
    np.savetxt(accuracy_file, np.array(val_acc))


def train(train_loader, model, fc, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    ratio = args.combine_ratio * (epoch / (training_configurations[args.model]['epochs']))
    # switch to train mode
    model.train()
    fc.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        target = target.cuda()
        x = x.cuda()
        input_var = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(target)

        # compute output
        loss, output = criterion(model, fc, input_var, target_var, ratio)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()


        if (i+1) % args.print_freq == 0:
            # print(discriminate_weights)
            fd = open(record_file, 'a+')
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1))

            print(string)
            # print(weights)
            fd.write(string + '\n')
            fd.close()


def validate(val_loader, model, fc, criterion, epoch, final_epoch=False):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(val_loader)

    # switch to evaluate mode
    model.eval()
    fc.eval()

    end = time.time()
    pred = []
    tar = []
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            features = model(input_var)
            output = fc(features)

        loss = criterion(output, target_var)
        _, p = output.topk(1, 1, True, True)
        pred.append(p.t())
        tar.append(target)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if (i+1) % args.print_freq == 0:
            fd = open(record_file, 'a+')
            string = ('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, (i+1), train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1))
            print(string)
            fd.write(string + '\n')
            fd.close()

    fd = open(record_file, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    print(string)
    fd.write(string + '\n')
    fd.close()
    val_acc.append(top1.ave)

    if final_epoch:
        global le
        # import ipdb;ipdb.set_trace()
        print(classification_report(tar[0].cpu().numpy(), pred[0].cpu().numpy()[0], target_names=le.classes_))
        print(confusion_matrix(tar[0].cpu().numpy(), pred[0].cpu().numpy()[0], labels=range(len(le.classes_))))
    return top1.ave

class Full_layer(torch.nn.Module):
    '''explicitly define the full connected layer'''

    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']

    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
