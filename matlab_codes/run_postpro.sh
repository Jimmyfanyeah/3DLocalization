set -x
# nSources=(5 10 15 20 30 40 50 60)
nSources=(5 10)
base_path="/home/lingjia/Documents/rpsf/temp/infer_result"
for nS in "${nSources[@]}"; do
    save_path="${base_path}/test${nS}"
    /home/lingjia/MATLAB/R2019a/bin/matlab \
        -nodisplay -nosplash -nodesktop \
        -r "nSource=${nS};pred_path='${base_path}/test${nS}';postpro;exit;"
done
set +x