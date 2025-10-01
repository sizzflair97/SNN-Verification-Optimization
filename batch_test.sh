export CUBLAS_WORKSPACE_CONFIG=:4096:8
trap "kill 0" SIGINT

encoding="latency" # "baseline" or "latency", only changes the name of the output log.
deltas=(3)
test_types=("mnist") # "mnist" "fmnist"
solvers=("np") # "np" "z3" "milp"
advs=("" "--adv") # "--adv" or ""
hidden_neurons=(512) # 128 256 384 512
num_steps=(64) # 64 128 192 256
repeat=1 # number of repetitions for each setting
for solver in ${solvers[@]}
do
  for hidden_neuron in ${hidden_neurons[@]}
  do
    for num_steps in ${num_steps[@]}
    do
      for delta in ${deltas[@]}
      do
        for test_type in ${test_types[@]}
        do
          for adv in "${advs[@]}"
          do
            script="python batch_test.py -p ${encoding} --delta-max ${delta} --test-type ${test_type} --${solver} --num-samples 14 --n-hidden-neurons ${hidden_neuron} --num-steps ${num_steps} ${adv} --repeat ${repeat}"
            echo $script
            $script &
          done
        done
        wait
      done
    done
  done
done