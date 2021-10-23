# 3D Localization of Point Sources

## Description
Following parts,
* Data info
* Network architecture
* Loss function
* Other info

## Data Info
* Input: 2D image with size 96X96
* Labels (ground-truth): 3D coordinates. 3D coordinates will be put into 3D grid and the value of that entry equals 1 (show existence) or flux value (also consider flux information) 

## Network Architecture

### Now
* cnn_residual
* cnn (initial cnn)

### Attempts before
* Initial cnn: DeepSTORM3D + dropout  
* cnn_no_dialte (0721): cnn without dilated conv layers, remove dilation rate `[2,4,8,16]`  
* cnn_ReLU (0808): `LeakyReLU` -> `ReLU`  
* cnn_concatIM (0808): remove features `torch.cat((out, im),1)`, which concatenate output of layers with original input image, used in DeepSTORM3D,
* cnn_residual (0808): the difference with concatIM as below,
    * `out = layer(out) + out` -> residual conv layer  
    * deconv1 and deconv2 layers with `+out` or not  
* cnn_duc (0808): more than 1 version
    * Loc3dResCNN: interpolate -> duc with aspp 
    * ResLocalizationCNN_DUC (0809): interpolate -> duc with plain conv, without leakyReLU and BN 
    * ResLocalizationCNN_DUC_v2 (0809): interpolate -> last layer duc 
* cnn_hdc (0810): more than 1 version  
    * ResLocalizationCNN_HDC: dilation `[1,1,2,4,8,16] -> [1,1,2,5,9,17]`
    * ResLocalizationCNN_HDC_v2 (v2): dilation changes `[1,1,2,4,8,16] -> 1,[1,2,5,9,17]*2`


## Loss function  
* MSE3D: Loss between output and ground-truth tensor. Entry values represent confidence or flux value. $MSE3D = ||G\otimes\hat y - G\otimes\hat y||^2$
* Dice loss: Dice loss only cares about the existence. $Dice=\frac{2|y\cap\hat y|}{|y|+|\hat y|}$
* Forward loss (0817): New fidelity term evaluate difference on 2D image, $Forward=||A\otimes\hat y-I_0||^2$. $A$ is the discrete 3D PSF matrix, $I_0$ is 2d observed image.
* Implementation of forward loss (0830): Implement forward loss (new fidelity term) by fft, details can be found in [GitHub](https://github.com/fkodom/fft-conv-pytorch) and [ZhiHu](https://zhuanlan.zhihu.com/p/300603589).
* Implementation of forward loss (0930): Implement fft with Neumann boundary condition, refered to [paper](https://epubs.siam.org/doi/pdf/10.1137/S1064827598341384).


## Other info
* [ReadME template](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
