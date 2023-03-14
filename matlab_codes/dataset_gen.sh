# set -x
# # date=$(date '+%Y-%m-%d-%H-%M-%S')
# base_path='/home/lingjia/Documents/rpsf/temp/20230314_poisson_#50_pt5L7'
# noise_type="poisson"
# Nindex_end=20
# nS=0
# /home/lingjia/MATLAB/R2019a/bin/matlab \
#     -nodisplay -nosplash -nodesktop \
#     -r "base_path='${base_path}';noise_type='${noise_type}';Nindex_end=${Nindex_end};nS=${nS};dataset_gen;exit;" 
# set +x


# testset
set -x
# nSources=('5' '10' '15' '20' '30' '40' '50' '60')
nSources=(5 10 15 20 30 40 50 60)
base_path="/home/lingjia/Documents/rpsf/temp/20230308_poisson_test_pt5L7"
noise_type="poisson"
Nindex_end=20
for nS in "${nSources[@]}"; do
    save_path="${base_path}/test${nS}"
    /home/lingjia/MATLAB/R2019a/bin/matlab \
        -nodisplay -nosplash -nodesktop \
        -r "base_path='${save_path}';noise_type='${noise_type}';Nindex_end=${Nindex_end};nS=${nS};dataset_gen;exit;"
done
set +x