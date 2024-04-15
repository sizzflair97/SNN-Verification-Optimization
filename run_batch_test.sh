#!/bin/bash

NP=(0 2)
M=(0)
# SEED=(42 617 133 1218)
# SEED=(42 617 133 1218 2016 1208 16384 14222)
SEED=(768 769 770 771 772 773 774 775)
for np in "${NP[@]}"
do
for memory in "${M[@]}"
do
for seed in "${SEED[@]}"
do
sbatch --export=ALL,np="$np",memory="$memory",seed="$seed",delta_max=1 batch_test.sbatch
done
done
done