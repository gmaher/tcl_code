#!/bin/bash
#SBATCH --job-name="i2inet"
#SBATCH --output="/home/gdmaher/i2inet.out"
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --mem=40000
cd /home/gdmaher/tcl_code
pwd
echo "START"
#rm -rf tempdir
python I2INet3DMed.py
#cp -rf /scratch/$USER/$SLURM_JOB_ID/* /home/gdmaher/copy
echo "DONE"

