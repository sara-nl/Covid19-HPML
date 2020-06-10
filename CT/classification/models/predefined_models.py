import torchxrayvision as xrv
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


def load_baseline(n_classes=2):
    """
    Baseline DenseNet model from this GitHub: https://github.com/UCSD-AI4H/COVID-CT/blob/master/CT_predict.py
    """
    model = xrv.models.DenseNet(num_classes=n_classes, in_channels=3)
    model.n_classes = n_classes
    return model


def load_densenet169(n_classes=2):
    model = models.densenet169(pretrained=False, num_classes=n_classes)
    return model


def load_densenet121(n_classes=2):
    model = models.densenet121(pretrained=False, num_classes=n_classes)
    return model


def load_densenet201(n_classes=2):
    model = models.densenet201(pretrained=False, num_classes=n_classes)
    return model


def load_efficientnet(n_classes=2):
    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=n_classes)
    return model
