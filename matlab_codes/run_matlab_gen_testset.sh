set -x
cd ./matlab_codes
nSources=(5 10 15 20 30 40 50 60)
# nSources=(5)
ntest=100
noise_type='gaussian'
save_path_base='/media/hdd/lingjia/hdd_rpsf/nonconvex_loss/data/gaussian_test'
for nSource in "${nSources[@]}"; do
    /home/lingjia/MATLAB/R2019a/bin/matlab \
        -nodisplay -nosplash -nodesktop \
        -r "nSource='${nSource}',N_test='${ntest}',noise_type='${noise_type}',save_path='${save_path_base}/test${nSource}';testset_gen;quit"
done
set +x
