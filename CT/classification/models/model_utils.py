from models.covidnet import CovidNet
from models.predefined_models import load_baseline


def get_model(opts):
    """ Load the model depending on the specified options """

    if opts.model == 'dense':
        model = load_baseline(n_classes=opts.n_classes)

    elif opts.model == 'covidnet_small':
        model = CovidNet(opts, model='small')

    elif opts.model == 'covidnet_large':
        model = CovidNet(opts, model='large')

    else:
        raise NotImplementedError

    model.n_classes = opts.n_classes
    return model
