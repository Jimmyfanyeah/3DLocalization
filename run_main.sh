set -x
nohup python3 main.py           --train_or_test='train'  \
                                --gpu_number='0'  \
                                --H=96  \
                                --W=96  \
                                --zeta=[-20,20]  \
                                --clear_dist=0  \
                                --D=250  \
                                --scaling_factor=170  \
                                --upsampling_factor=2  \
                                --model_use='cnn_residual'  \
                                --batch_size=8  \
                                --initial_learning_rate=0.0005  \
                                --lr_decay_per_epoch=3  \
                                --lr_decay_factor=0.5  \
                                --max_epoch=1  \
                                --save_epoch=10  \
                                --data_path='/home/lingjia/Documents/3dloc_data/train/0620_uniformFlux'  \
                                --save_path='/home/lingjia/Documents/3dloc_result/CNN_v2'  \
                                > /home/lingjia/Documents/3dloc_result/CNN_v2/log/1001.log 2>&1 &
set +x