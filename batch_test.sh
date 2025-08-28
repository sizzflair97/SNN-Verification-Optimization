export CUBLAS_WORKSPACE_CONFIG=:4096:8

encoding="latency"
deltas=(1 2 3)
test_types=("mnist" "fmnist")
# solvers=("np" "z3" "milp")
solvers=("milp")
hidden_neurons=(10 12 14 16 18 20)
for hidden_neuron in ${hidden_neurons[@]}
do
  for delta in ${deltas[@]}
  do
    for test_type in ${test_types[@]}
    do
      for solver in ${solvers[@]}
        do
        python batch_test.py -p ${encoding} --delta-max ${delta} --test-type ${test_type} --${solver} \
        --num-samples 14 --n-hidden-neurons ${hidden_neuron}
        done
    done
  done
done