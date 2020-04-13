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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['plateau', 'step'])
    parser.add_argument('--patience', type=int, default=9)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--save-model', type=bool, default=True)

    parser.add_argument('--run_name', type=str, default='run')

    opts = parser.parse_args()

    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"

    return opts
