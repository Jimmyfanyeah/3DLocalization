source /home/lingjia/.bashrc
source activate env_deepstorm3d
set -x
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
log_comment="test_temp"
name_time=$(date '+%Y-%m-%d-%H-%M-%S')
nSources=('5' '10' '15' '20' '30' '40' '50' '60')
model_path='/home/lingjia/Documents/rpsf/temp/trained_model/locneths_122912-nTrain9891-lr0.001-Epoch190-batchSize10-D250-cnn_residual/checkpoint_best_loss'
log_name="/home/lingjia/Documents/rpsf/temp/infer_result/${name_time}_${log_comment}_test.out"
printf "Name Time: ${name_time}\n" >> ${log_name}
printf "Model Path: ${model_path}\n" >> ${log_name}
printf "Log Comment: ${log_comment}\n" >> ${log_name}
for nSource in "${nSources[@]}"; do
    printf "\n\n======= nSource ${nSource} =======\n" >> ${log_name}
    python3 main.py     --train_or_test='test'  \
                        --gpu_number='1' \
                        --H=96  \
                        --W=96  \
                        --zmax=20  \
                        --clear_dist=0  \
                        --D=250  \
                        --scaling_factor=800  \
                        --upsampling_factor=2  \
                        --model_use='cnn_residual'  \
                        --batch_size=1  \
                        --checkpoint_path=${model_path} \
                        --data_path="/home/lingjia/Documents/rpsf/temp/20230308_poisson_test_pt5L7/test${nSource}" \
                        --save_path="/home/lingjia/Documents/rpsf/temp/infer_result/test${nSource}" \
                        >> ${log_name}
done
printf "End: `date`\n\n\n" >> ${log_name}
set +x

