import argparse
import time
import os


def parse_arguments():
    """Argument Parser for the commandline argments
    :returns: command line arguments
    """
    ##########################################################################
    #                            Training setting                            #
    ##########################################################################
    parser = argparse.ArgumentParser()

    # Training options
    parser.add_argument('--model', type=str, default='covidnet_large')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes in the classification problem')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--img_size', type=int, default=512, help='Input image size')
    parser.add_argument('--data_root', type=str, default=os.path.join('data', 'train'))
    parser.add_argument('--folder1', type=str, default='brazilian')
    parser.add_argument('--folder2', type=str, default='zenodo')
    parser.add_argument('--folder3', type=str, default='medical_segmentation')

    # Learning rate scheduler options
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['plateau', 'step', 'cosine'])
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=0.8, help='Gamma for StepLR')

    # Logging options
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--run_name', type=str, default='run')

    # Other options
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2)

    opts = parser.parse_args()

    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"

    return opts
