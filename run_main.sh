set -x
nohup python3 main.py           --gpu_number='0,1'  \
                                --zmax=20  \
                                --D=250  \
                                --clear_dist=1  \
                                --upsampling_factor=2  \
                                --scaling_factor=170  \
                                --batch_size=8  \
                                --initial_learning_rate=0.0005  \
                                --lr_decay_per_epoch=3  \
                                --lr_decay_factor=0.5  \
                                --saveEpoch=10  \
                                --maxEpoch=190  \
                                --train_or_test='train'  \
                                --training_data_path='../data_train'  \
                                --result_path='../trained_model'  \
                                --resume_training=0  \
                                --checkpoint_path=''  \
                                --model_use='cnn_residual'  \
                                > ../trained_model/training_main_sh.log 2>&1 &
set +x