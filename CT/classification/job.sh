#!/bin/bash
#Set job requirements
#SBATCH -n 12
#SBATCH -t 2-00:00:00
#SBATCH -p gpu_titanrtx_shared

module use ~/environment-modules-lisa
module load 2020
module load TensorFlow/2.1.0-foss-2019b-Python-3.7.4-CUDA-10.1.243

conda deactivate

VIRTENV=covid19_classification
VIRTENV_ROOT=~/.virtualenvs

clear
source $VIRTENV_ROOT/$VIRTENV/bin/activate

python3 main.py --batch_size 64