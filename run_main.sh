set -x
nohup python3 ./NN/main.py       --device='cuda:1' \
                            --param_file='/home/lingjia/Documents/rpsf/NN/param.yaml'\
                            --data_path_override='/home/lingjia/Documents/rpsf/decode_variant/data_train/10k_pt2L5'\
                            --img_size_override=96 \
                            --log_comment='10k_pt2L5_decode_variant_v1' \
                            > ./decode_variant/run_log/training_10k_pt50L5_1.log 2>&1 &
set +x
