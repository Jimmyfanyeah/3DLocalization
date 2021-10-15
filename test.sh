set -x
python3 main.py      --gpu_number='1' \
    --checkpoint_path='../trained_model/0808-nTrain9000-lr0.0007-Epoch100-batchSize8-D250-cnn_residual/checkpoint_best_loss' \
            --training_data_path='../data_test/test35' \
            --result_path='../test_output/test35' \
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