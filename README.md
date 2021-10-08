# 3D Localization of Point Sources

## Description

### Code info
- __utils/calc_metrics.py__: evaluate on original prediction (precision & recall)
* __utils/cnn.py__: two network architecture. Main difference: residual layer or not
    - LocalizationCNN
    - ResLocalizationCNN
* __utils/data.py__: build data generator. Main difference: line 176-180, load 2d ground-truth image or not. Modify the choice in line 97-101
    - ImagesDataset: used without forward loss 
    - ImagesDataset_v2 & ImagesDataset_test: used with forward loss. 
* __utils/fft_conv.py__: use FFT to implement 3d convolution when calculating forward loss
* __utils/helper.py__: some auxiliary functions. Function buildModel (line 75-81) choice the model to use based on argument
* __utils/loss.py__: loss functions include 
    - dice (not use now) 
    - regularization term: same as Chao's paper 
    - mse3d: ||G*y^-G*y||^2 
    - mse2d (forward loss): ||A*y^-y||^2. A is 3D PSF matrix. The final criterion in calculate_loss_v2 (line 140)
* __utils/postprocess.py__: post-processing / save test prediction
    - Postprocess: cluster + thresh, same setting as DeepSTORM3D
    - Postprocess__v0: store points with conf>0
* __utils/test_model.py__: test
* __utils/train_model.py__: train
* __utils/lr_find.py__: find suitbal initial learning rate
* __utils/main.py__: main part


### Models
Find previous models [here](https://drive.google.com/drive/folders/1Z-UTRqAauBXRbDDDtC_uCLNeBw6O5bY6?usp=sharing) or server(91)/tmp

