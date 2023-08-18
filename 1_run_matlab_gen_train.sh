set -x
/home/lingjia/MATLAB/R2019a/bin/matlab \
    -nodisplay -nosplash -nodesktop \
    -r "run('./matlab_codes/Generate_training_images.m');exit;" 
> ../test_output/run_matlab_gen_train.log 2>&1 &
set +x