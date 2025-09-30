export CUBLAS_WORKSPACE_CONFIG=:4096:8

encoding="latency"
delta=1
test_type="mnist"
# solvers=("np" "z3" "milp")
solver="np"
adv="--adv"
hidden_neuron=128
num_steps=256
repeat=20
python batch_test.py -p ${encoding} --delta-max ${delta} --test-type ${test_type} --${solver} --num-samples 14 --n-hidden-neurons ${hidden_neuron} --num-steps ${num_steps} ${adv} --repeat ${repeat}