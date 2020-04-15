import torchxrayvision as xrv


def load_baseline(n_classes=2):
    """
    Baseline DenseNet model from this GitHub: https://github.com/UCSD-AI4H/COVID-CT/blob/master/CT_predict.py
    """
    model = xrv.models.DenseNet(num_classes=n_classes, in_channels=3)
    model.n_classes = n_classes
    return model
