#!/bin/sh
set -x
cd ./matlab_codes
pred_path='/media/hdd/lingjia/hdd_rpsf/nonconvex_loss/result/cel0_D250/2022-10-22-14-14-31-w1_1-mse3d_cel0-1e-5-D250'
date=$(date '+%Y-%m-%d-%H-%M-%S')
/home/lingjia/MATLAB/R2019a/bin/matlab \
    -nodisplay -nosplash -nodesktop \
    -r "pred_path_base='${pred_path}';postpro;quit"
set +x
