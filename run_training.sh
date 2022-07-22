#!/bin/bash
set -x
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# declare -a exps=("exp1" "exp2" "exp3" "exp4" "exp5")
# tss=(10 20 50 80 100)
datenow=$(date '+%Y-%m-%d-%H-%M-%S')
logcomment=$"30k_pt50L5_decode_variant_v2_attempt1"

echo -e "Begin:" `date` > ../output_log/2_run_training_$logcomment.log
echo "Log comment: $logcomment" >> ../output_log/2_run_training_$logcomment.log
python3 main.py     --device='cuda:1'  \
                    --param_file='./param.yaml'  \
                    --data_path_override='/media/hdd/lingjia/hdd_rpsf/data/plain/train/0620_uniformFlux'  \
                    --img_size_override=96  \
                    --log_comment=$logcomment
>> ../output_log/2_run_training_$logcomment.log 2>&1

echo -e "End:" `date` >>../output_log/2_run_training_$logcomment.log
set +x