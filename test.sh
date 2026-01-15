export CUBLAS_WORKSPACE_CONFIG=:4096:8

encoding="latency"
delta=2
test_type="mnist"
manual_indices=(58469 )
# solvers=("np" "z3" "milp")
solver="np"
# psm=("--psm" "")
psm=""
adv=""
hidden_neuron=512
num_steps=64
repeat=1
python batch_test.py -p ${encoding} --delta-max ${delta} --test-type ${test_type} --${solver} ${psm} ${adv} --num-samples 14 --n-hidden-neurons ${hidden_neuron} --num-steps ${num_steps} --repeat ${repeat} --manual-indices ${manual_indices[@]}  