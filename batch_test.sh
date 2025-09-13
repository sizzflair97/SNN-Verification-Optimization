export CUBLAS_WORKSPACE_CONFIG=:4096:8

encoding="latency"
deltas=(2)
test_types=("mnist")
# solvers=("np" "z3" "milp")
solvers=("np")
adv="--adv"
hidden_neurons=(128 256 384 512)
num_steps=(256)
for hidden_neuron in ${hidden_neurons[@]}
do
  for num_steps in ${num_steps[@]}
  do
    for delta in ${deltas[@]}
    do
      for test_type in ${test_types[@]}
      do
        for solver in ${solvers[@]}
          do
          python batch_test.py -p ${encoding} --delta-max ${delta} --test-type ${test_type} --${solver} \
          --num-samples 14 --n-hidden-neurons ${hidden_neuron} --num-steps ${num_steps}
          done
      done
    done
  done
done