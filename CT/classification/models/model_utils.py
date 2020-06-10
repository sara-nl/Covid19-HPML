from models.covidnet import CovidNet
from models.predefined_models import load_baseline, load_densenet169, load_densenet121, load_densenet201
from efficientnet_pytorch import EfficientNet


def get_model(opts):
    """ Load the model depending on the specified options """

    if opts.model == 'dense-169':
        model = load_densenet169(opts.n_classes).cpu()

    elif opts.model == 'dense-121':
        model = load_densenet121(opts.n_classes).cpu()

    elif opts.model == 'dense-201':
        model = load_densenet201(opts.n_classes).cpu()

    elif opts.model == 'covidnet-small':
        model = CovidNet(opts, model='small').cpu()

    elif opts.model == 'covidnet-large':
        model = CovidNet(opts, model='large').cpu()

    elif 'efficientnet' in opts.model:
        model = EfficientNet.from_name(opts.model,
                                       override_params={"num_classes": opts.n_classes, 'image_size': opts.img_size})

    else:
        raise NotImplementedError

    model.n_classes = opts.n_classes
    return model
