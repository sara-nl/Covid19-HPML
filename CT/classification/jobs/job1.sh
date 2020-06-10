#!/bin/bash
#Set job requirements
#SBATCH -t 5-00:00:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx

module purge
module load 2019
module load Python/3.6.6-foss-2019b
module load CUDA/10.1.243

conda deactivate

VIRTENV=covid19_classification
VIRTENV_ROOT=~/.virtualenvs

clear
source $VIRTENV_ROOT/$VIRTENV/bin/activate

cd ..

python3 main.py --batch_size 32 --model efficientnet-b4 --lr_scheduler plateau --run_name b4
#python3 main.py --batch_size 32 --model efficientnet-b5 --lr_scheduler plateau --run_name b5
#python3 main.py --batch_size 32 --model efficientnet-b6 --lr_scheduler plateau --run_name b6
#python3 main.py --batch_size 32 --model efficientnet-b7 --lr_scheduler plateau --run_name b7
#python3 main.py --batch_size 32 --model dense-201 --lr_scheduler plateau --run_name dense201


