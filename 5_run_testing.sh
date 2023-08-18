#!/bin/bash
set -x
declare -a cps=("081523-nTrain9000-lr0.001-Epoch190-batchSize18-D250-cnn_residual/checkpoint_best_loss")

pts=(5 10 15 20 25 30 35 40 45)
for cp in "${cps[@]}"; do
    echo "$cp" >> ../test_output/postpro_result.csv
    echo "batch,testsize,nSource,id,recall" >> ../data_train/hardsamples/summary.csv
    for pt in ${pts[*]}; do
        python3 main.py      \
            --gpu_number='1' \
            --checkpoint_path='../../trained_model/'$cp \
            --training_data_path='../data_test/test'$pt \
            --result_path='../test_output/test'$pt \
            --model_use='cnn_residual'  \
            --post_pro=0  \
            --D=250  \
            --zmax=20  \
            --clear_dist=1  \
            --upsampling_factor=2  \
            --scaling_factor=170  \
            --batch_size=8  \
            --train_or_test='test' 

        /home/lingjia/MATLAB/R2019a/bin/matlab -nodisplay -nosplash -nodesktop \
            -r "nSource = $pt;hs_recall_bar=0.95;run('./matlab_codes/postpro_loc_batch.m');exit;" 
    done
done
set +x