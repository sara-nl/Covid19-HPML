import argparse
import time


def parse_arguments():
    """Argument Parser for the commandline argments
    :returns: command line arguments
    """
    ##########################################################################
    #                            Training setting                            #
    ##########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--batch_size', type=int, default=8)

    # Model loading parameters
    parser.add_argument('--model', type=str, choices=['unet', 'fcn', 'deeplab', 'deeplabv3+'])
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101', 'xception'])

    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['plateau', 'step'])
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.9, help='Gamma to use in step LR scheduler')

    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--run_name', type=str, default='run')

    opts = parser.parse_args()

    if opts.model in ['fcn', 'deeplab', 'deeplabv3+']:
        opts.run_name = f"{opts.run_name}_{opts.model}_{opts.backbone}_{time.strftime('%Y%m%dT%H%M%S')}"
    else:
        opts.run_name = f"{opts.run_name}_{opts.model}_{time.strftime('%Y%m%dT%H%M%S')}"

    return opts
