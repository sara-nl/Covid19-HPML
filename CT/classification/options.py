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
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--img_size', type=int, default=512, help='Input image size')
    parser.add_argument('--data_root', type=str, default=os.path.join('data', 'train'))
    parser.add_argument('--folder1', type=str, default='brazilian')
    parser.add_argument('--folder2', type=str, default='zenodo')
    parser.add_argument('--folder3', type=str, default='medical_segmentation')
    parser.add_argument('--train_mode', type=str, choices=['combined', 'oversampling', 'pretrain_and_finetune'],
                        help="How to perform training. 'Combined' uses all datasets from 'folder1', 'folder2', "
                             "'folder3' (if given), while 'oversampling' oversamples the dataset from 'folder1'. "
                             "Pretraining and then finetuning is also possible.")
    parser.add_argument('--finetune_epoch', type=int, default=None,
                        help='After this number of epoch finetuning will start')

    # Learning rate scheduler options
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['plateau', 'step', 'cosine'])
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=0.8, help='Gamma for StepLR')
    parser.add_argument('--step_size', type=int, default=3, help='Step size of StepLR')

    # Logging options
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--run_name', type=str, default='run')

    # Other options
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2)
    parser.add_argument('--no_data_parallel', action='store_true', default=False)

    opts = parser.parse_args()

    # Add other options
    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    opts.eval_batch_size = opts.batch_size * 2

    # Perform checks
    if opts.train_mode == 'pretrain_and_finetune':
        assert opts.finetune_epoch is not None, "You should specify then to start finetuning"

    return opts
