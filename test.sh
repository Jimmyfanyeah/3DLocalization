set -x
python3 main.py      --gpu_number='0' \
    --checkpoint_path='/home/lingjia/Documents/3dloc_result/CNN_v2/0930-nTrain9000-lr0.0005-Epoch199-batchSize8-D250-cnn_residual/checkpoint_best_loss' \
            --training_data_path='/home/lingjia/Documents/3dloc_data/special_case' \
            --result_path='/home/lingjia/Documents/3dloc_result/CNN_v2/TestResult/0930/special_case' \
            --test_id_loc='/home/lingjia/Documents/3dloc_result/data/id1.txt' \
            --model_use='cnn_residual'  \
            --post_pro=0  \
            --D=250  \
            --zmax=20  \
            --clear_dist=1  \
            --upsampling_factor=2  \
            --scaling_factor=170  \
            --batch_size=8  \
            --train_or_test='test' 
set +x

# /home/lingjia/Documents/3dloc_result/CNN/0808-nTrain9000-lr0.0007-Epoch100-batchSize8-D250-cnn_residual/checkpoint_best_loss
# /home/lingjia/Documents/3dloc_result/CNN_v2/0909-nTrain9000-lr0.0005-Epoch300-batchSize8-D250-cnn_residual/checkpoint_best_loss