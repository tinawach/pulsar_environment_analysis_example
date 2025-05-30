#!/bin/bash -l
#SBATCH --job-name=dataset1
#SBATCH --time=23:00:00
#SBATCH --mem=10G
#SBATCH -o outfile1  # send stdout to outfile

conda activate gammapy-1.20
python dataset1.py