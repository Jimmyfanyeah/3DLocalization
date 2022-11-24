set -x
cd ./matlab_codes
date=$(date '+%Y-%m-%d-%H-%M-%S')
base_path='/media/hdd/lingjia/hdd_rpsf/nonconvex_loss/temp/data_gen_temp'
noise_type="poisson";
/home/lingjia/MATLAB/R2019a/bin/matlab \
    -nodisplay -nosplash -nodesktop \
    -r "base_path='${base_path}',noise_type='${noise_type}';trainset_gen;exit;" 
set +x