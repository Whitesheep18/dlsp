#!/bin/bash

#BSUB -q gpua100 
#BSUB -J train
#BSUB -o outs/train_%J.out
#BSUB -n 4
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 1:00
#BSUB -gpu "num=1:mode=exclusive_process"

module load python3/3.11.3
module load cuda/12.1.1
source ~/Agar/venv3/bin/activate

JID=${LSB_JOBID}
python train.py --architecture lstm --initialization He --epochs 500 --device cuda --fig_path figures/diagnostics --plot_every_n_epochs 50 --dropout 0.1 --tag dropout