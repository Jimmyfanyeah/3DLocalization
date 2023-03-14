#!/bin/bash
source /home/lingjia/.bashrc
source activate env_deepstorm3d
set -x
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
log_comment=$"train_demo"
name_time=$(date '+%Y-%m-%d-%H-%M-%S')
# name_time="9000-00"
log_name="/home/lingjia/Documents/rpsf/temp/trained_model/${name_time}_${log_comment}_train.log"
printf "Start Time: `date`\nName Time: ${name_time}\nLog Comment: ${log_comment}\n" >> ${log_name}
python3 main.py         --train_or_test='train'  \
                        --gpu_number='1'  \
                        --num_im=20  \
                        --H=96  \
                        --W=96  \
                        --zmax=20  \
                        --clear_dist=0  \
                        --D=250  \
                        --scaling_factor=800  \
                        --upsampling_factor=2  \
                        --model_use='cnn_residual'  \
                        --batch_size=4  \
                        --initial_learning_rate=6e-4  \
                        --lr_decay_per_epoch=3  \
                        --lr_decay_factor=0.5  \
                        --max_epoch=5  \
                        --save_epoch=2  \
                        --data_path='/home/lingjia/Documents/rpsf/temp/20230308_poisson_#50_pt5L7'  \
                        --save_path='/home/lingjia/Documents/rpsf/temp/trained_model'  \
                        >> ${log_name}
    printf "End: `date`\n\n\n" >>${log_name}
set +x
