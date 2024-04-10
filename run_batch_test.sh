#!/bin/bash

D=("" "-d")
M=(0)
SEED=(42 617 133 1218 2016 1208 16384 14222)
for dnp in "${D[@]}"
do
for memory in "${M[@]}"
do
for seed in "${SEED[@]}"
do
sbatch --export=ALL,dnp="$dnp",memory="$memory",seed="$seed" batch_test.sbatch
done
done
done