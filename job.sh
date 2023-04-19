#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem 128000
#SBATCH -t 3-00:00
#SBATCH --open-mode=append
#SBATCH -o jobIO.out
#SBATCH -e jobE.err
module load python
module load Anaconda3/2020.11
module load cuda/9.0-fasrc02 cudnn/7.4.1.5_cuda9.0-fasrc01
source activate flatland
#apt-get remove libtcmalloc*
python3 train.py --env starcraft --n_workers 1  --env_name 2s_vs_1sc