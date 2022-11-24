source /home/lingjia/.bashrc
source activate deepstorm3d
set -x
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
log_comment=$"1e-5-D250"
name_time=$(date '+%Y-%m-%d-%H-%M-%S')
nSources=('5' '10' '15' '20' '30' '40' '50' '60')
model_path='/media/hdd/lingjia/hdd_rpsf/nonconvex_loss/trained_model/cel0_D250/2022-10-22-14-14-31-lr0.001-bs16-D250-Ep200-nT9000-w1_1-mse3d_cel0-1e-05-LocNet/ckpt_best_loss'
log_name='/media/hdd/lingjia/hdd_rpsf/nonconvex_loss/log/'${name_time}'_'${log_comment}'_test.log'
printf "Name Time: ${name_time}\n" >> ${log_name}
printf "Model Path: ${model_path}\n" >> ${log_name}
printf "Log Comment: ${log_comment}\n" >> ${log_name}
for nSource in "${nSources[@]}"; do
    python3 main.py     --train_or_test='test'  \
                        --gpu_number='2'  \
                        --H=96  \
                        --W=96  \
                        --zmin=-20  \
                        --zmax=20  \
                        --clear_dist=1  \
                        --D=250  \
                        --scaling_factor=800  \
                        --upsampling_factor=2  \
                        --model_use='LocNet'  \
                        --batch_size=16  \
                        --checkpoint_path=${model_path}  \
                        --data_path="/media/hdd/lingjia/hdd_rpsf/nonconvex_loss/data/gaussian_test/test${nSource}"  \
                        --save_path='/media/hdd/lingjia/hdd_rpsf/nonconvex_loss/result/cel0_D250'  \
                        --port='123789'  \
                        --weight='1_1'  \
                        --extra_loss='mse3d_cel0'  \
                        --klnc_a=1e5  \
                        --cel0_mu=1e-5  \
                        --log_comment=${log_comment}  \
                        >> ${log_name}
done
printf "End: `date`\n\n\n" >> ${log_name}
set +x