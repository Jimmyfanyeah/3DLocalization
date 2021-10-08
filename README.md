# 3D Localization of Point Sources

## Description

### Code info
* __utils/calc_metrics.py__: evaluate on original prediction (precision & recall)
* __utils/cnn.py__: two network architecture, (1) LocalizationCNN (2) ResLocalizationCNN. Main difference: residual layer or not
* __utils/data.py__: (1) ImagesDataset: used without forward loss (2) ImagesDataset_v2 & ImagesDataset_test: used with forward loss. Main difference: line 176-180, load 2d ground-truth image or not. Modify the choice in line 97-101
* __utils/fft_conv.py__: use FFT to implement 3d convolution when calculating forward loss
* __utils/helper.py__: some auxiliary functions. Fun ```buildModel``` (line 75-81) choice the model to use based on argument
* __utils/loss.py__: loss functions include (1) dice (not use now) (2) regularization term: same as Chao's paper (3) mse3d: ||G*y^-G*y||^2 (4) mse2d (forward loss): ||A*y^-y||^2. A is 3D PSF matrix. The final criterion in calculate_loss_v2 (line 140)
* __utils/postprocess.py__: (1) ```Postprocess```: cluster + thresh, same setting as DeepSTORM3D (2) ```Postprocess__v0```: store points with conf>0
* __utils/test_model.py__: test
* __utils/train_model.py__: train
* __utils/lr_find.py__: find suitbal initial learning rate
* __utils/main.py__: main part


### Models
Find previous models [here](https://drive.google.com/drive/folders/1Z-UTRqAauBXRbDDDtC_uCLNeBw6O5bY6?usp=sharing) or server(91)/tmp

