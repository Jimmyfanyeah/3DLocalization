source /home/lingjia/.bashrc
source activate deepstorm3d
set -x
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
log_comment=$"mse3d"
name_time=$(date '+%Y-%m-%d-%H-%M-%S')
log_name='/home/lingjia/Documents/rpsf/finetune_loss/temp/log/'${name_time}_${log_comment}'_train.log'
printf "Name Time: ${name_time}\nStart Time: `date`\nLog Comment: ${log_comment}\nweight: ${weight}\n" >> ${log_name}
python3 main.py     --train_or_test='train'  \
                    --gpu_number='1' \
                    --name_time=${name_time}  \
                    --num_im=10000  \
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
                    --initial_learning_rate=1e-3  \
                    --lr_decay_per_epoch=7  \
                    --lr_decay_factor=0.5  \
                    --max_epoch=30  \
                    --save_epoch=10  \
                    --data_path='/media/hdd/lingjia/hdd_rpsf/20220917_nonconvex_loss/data/gaussian_10k_pt50L5'  \
                    --save_path='/home/lingjia/Documents/rpsf/finetune_loss/temp/trained_model'  \
                    --port='123457'  \
                    --weight='1_0_0'  \
                    --extra_loss='mse3d_klnc_forward'  \
                    --cel0_mu=1  \
                    --klnc_a=10 \
                    >> ${log_name}
printf "End: `date`\n\n\n" >>${log_name}
set +x