import torchvision


def load_deeplab_resnet101(n_classes):
    """ Load the Deeplab V3 with ResNet 101 backbone model """
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=n_classes)
    return model


def load_deeplab_resnet50(n_classes):
    """ Load the Deeplab V3 with ResNet 50 backbone model """
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=n_classes)
    return model


def load_fcn_resnet50(n_classes):
    """ Load the fully convolutional model with ResNet 50 backbone """
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=n_classes)
    return model


def load_fcn_resnet101(n_classes):
    """ Load the fully convolutional model with ResNet 50 backbone """
    model = torchvision.models.segmentation.fcn_resnet101(pretrained=False, num_classes=n_classes)
    return model