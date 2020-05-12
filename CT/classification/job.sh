#!/bin/bash
#Set job requirements
#SBATCH -t 2-00:00:00
#SBATCH -p gpu_titanrtx

module use ~/environment-modules-lisa
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243

conda deactivate

VIRTENV=covid19_classification
VIRTENV_ROOT=~/.virtualenvs

clear
source $VIRTENV_ROOT/$VIRTENV/bin/activate

# python3 main.py --batch_size 16 --model dense --lr_scheduler step --img_size 512 --run_name dense_512
# python3 main.py --batch_size 16 --model covidnet_large --lr_scheduler step --img_size 512 --run_name large_512
# python3 main.py --batch_size 16 --model covidnet_small --lr_scheduler step --img_size 512 --run_name small_512
#
# python3 main.py --batch_size 16 --model dense --lr_scheduler step --img_size 299 --run_name dense_299
# python3 main.py --batch_size 16 --model covidnet_large --lr_scheduler step --img_size 299 --run_name large_299
python3 main.py --batch_size 16 --model covidnet_small --lr_scheduler step --img_size 299 --run_name small_299


