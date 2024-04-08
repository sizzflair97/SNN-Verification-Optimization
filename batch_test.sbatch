#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --partition=cpu1
#SBATCH --cpus-per-task=8
#SBATCH -o log/%N.%j.out 
#SBATCH -e log/%N.%j.err 

source ~/miniconda3/bin/activate smt
prmpt="python3 batch_test.py $dnp --reuse_level $memory --seed $seed -p ${SLURM_JOB_ID}"
echo "$prmpt"
$prmpt