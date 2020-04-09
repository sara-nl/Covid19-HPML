from models.torchvision_models import *
from models.unet import UNet
from models.deeplab_resnet import DeepLabv3_plus as DeepLabv3_plus_resnet
from models.deeplab_xception import DeepLabv3_plus as DeepLabv3_plus_xception


def load_model(opts, n_classes=4):
    if opts.model == 'unet':
        model = UNet(n_classes=n_classes)
    elif opts.model == 'fcn':
        if opts.backbone == 'resnet50':
            model = load_fcn_resnet50(n_classes)
        elif opts.backbone == 'resnet101':
            model = load_fcn_resnet101(n_classes)
        else:
            raise NotImplementedError("Invalid backbone specified")
    elif opts.model == 'deeplab':
        if opts.backbone == 'resnet50':
            model = load_deeplab_resnet50(n_classes)
        elif opts.backbone == 'resnet101':
            model = load_deeplab_resnet101(n_classes)
        else:
            raise NotImplementedError("Invalid backbone specified")
    elif opts.model == 'deeplabv3+':
        if opts.backbone == 'resnet101':
            model = DeepLabv3_plus_resnet(n_classes)
        elif opts.backbone == 'xception':
            model = DeepLabv3_plus_xception(n_classes)
        else:
            raise NotImplementedError("Invalid backbone specified")
    else:
        raise NotImplementedError("Invalid model type specified")

    model.n_classes = n_classes
    return model
