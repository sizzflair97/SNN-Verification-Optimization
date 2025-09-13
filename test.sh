export CUBLAS_WORKSPACE_CONFIG=:4096:8

encoding="latency"
delta=2
test_type="mnist"
# solvers=("np" "z3" "milp")
solver="np"
adv=""
hidden_neuron=512
python batch_test.py -p ${encoding} --delta-max ${delta} --test-type ${test_type} --${solver} \
    --num-samples 14 --n-hidden-neurons ${hidden_neuron} ${adv}