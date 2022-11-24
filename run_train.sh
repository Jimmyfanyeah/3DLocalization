#!/bin/bash
#SBATCH --job-name=nc
#SBATCH --nodes=1
#SBATCH --partition=gpu_7d1g
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/ljdai2/scratch/scratch_rpsf/nonconvex_loss/sbatch_output/klnc_locnet_v2.out

source /home/ljdai2/.bashrc
source activate deepstorm3d
set -x
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
log_comment=$"nc"
name_time=$(date '+%Y-%m-%d-%H-%M-%S')
log_name='/home/ljdai2/scratch/scratch_rpsf/nonconvex_loss/log/'${name_time}_${log_comment}'_train.log'
weights=(0.001 0.01 0.1 1 10 100)
for weight in "${weights[@]}"; do
    printf "Name Time: ${name_time}\nStart Time: `date`\nLog Comment: ${log_comment}\nweight: ${weight}\n" >> ${log_name}
    python3 main.py     --train_or_test='train'  \
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
                        --max_epoch=2  \
                        --save_epoch=10  \
                        --data_path='/home/ljdai2/scratch/scratch_rpsf/nonconvex_loss/data/gaussian_10k_pt50L5'  \
                        --save_path='/home/ljdai2/scratch/scratch_rpsf/nonconvex_loss/trained_model'  \
                        --port='123789'  \
                        --weight='1_1_1_1'  \
                        --extra_loss='mse3d_cel0_klnc_forward'  \
                        --cel0_mu=${weight}  \
                        --klnc_a=10 \
                        >> ${log_name}
    printf "End: `date`\n\n\n" >>${log_name}
done
set +x