#!/bin/bash

#SBATCH -J job_name
#SBATCH -p gpu
#SBATCH -A r00068
#SBATCH -o job_logs/filename_%j.txt
#SBATCH -e job_logs/filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=deduggi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node v100:4
#SBATCH --time=7:00:00
#SBATCH --mem=50gb

#Load any modules that your program needs
module load deeplearning/2.9.1

#Run your program
python train.py --name "tiim_$(date +%F_%H-%M-%S)" --val-interval 100