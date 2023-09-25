#!/bin/bash
#
#SBATCH -p gpu # partition (queue)
#SBATCH --exclude=gpu-sr670-20
#SBATCH -N 1   # number of nodes
#SBATCH --mem 120G # memory pool for all cores
#SBATCH --gres gpu:rtx5000:1
#SBATCH -t 15-0:0 # time (D-HH:MM)
#SBATCH -o lsdtrain.out
#SBATCH -e lsdtrain.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ucqfbfg@ucl.ac.uk


echo "loading minconda"
module load miniconda/4.9.2
echo "loading cuda"
module load cuda/12.0
echo "initializing environment"
echo "starting script"
python3 LSD_train.py