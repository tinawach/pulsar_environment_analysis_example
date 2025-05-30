#!/bin/bash -l
#SBATCH --job-name=dataset2
#SBATCH --time=23:50:00
#SBATCH --mem=10G
#SBATCH -o outfilegroup  # send stdout to outfile

conda activate gammapy-1.20
python grouping.py