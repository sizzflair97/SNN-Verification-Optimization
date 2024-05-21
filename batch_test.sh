export CUBLAS_WORKSPACE_CONFIG=:4096:8
# python batch_test.py -p Control --np_level 0 --reuse_level 0 --delta_max 1&\
 python batch_test.py -p DNP_RP --np_level 1 --reuse_level 0 --delta_max 1