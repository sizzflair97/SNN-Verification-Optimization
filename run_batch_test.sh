#!/bin/bash

D=("" "-d")
M=(0 2)
for iter in 0 1
do
for dnp in "${D[@]}"
do
for memory in "${M[@]}"
do
sbatch --export=ALL,dnp="$dnp",memory="$memory" batch_test.sbatch
done
done
done