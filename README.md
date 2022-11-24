# Finetune loss terms


## Use different loss terms and weights
### in `run_train.sh`,
- `extra_loss`: define which loss terms will be used, concatenate them use "_". choices:
    1. mse3d 
    2. cel0
    3. klnc (refers to the non convex termn in KLNC algorithm)
    4. forward (refers to MSE of 2D observed image)
- `weight`: weight of each loss term, the number of weights should be same with `extra_loss`.
- `cel0_mu`: value of $\mu$ in CEL0 loss
- `klnc_a`: value of $a$ in NC loss.=
```
--weight='1_1_1_1'  \
--extra_loss='mse3d_cel0_klnc_forward'  \
--cel0_mu=1  \
--klnc_a=10 \
```







