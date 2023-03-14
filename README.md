# 3D Localization of Point Sources
This branch shows a demo of running DeepSTORM3D type network on 3D localization of point sources using rotating PSF.

## Installation
Set up the environment for running codes using ANACONDA.

First clone the  repository
```
git clone -b 20230308_deepstorm3d_demo https://github.com/Jimmyfanyeah/3DLocalization.git
cd "20230308_deepstorm3d_demo"
```

Then start a virtual environment with new environment variables
```
conda env create -f env_deepstorm3d.yml
conda activate env_deepstorm3d
```

## Executing program
The whole framework contains three parts: 
1. Generate datasets with labels 
2. Train and infer on datasets 
3. Post-process and evaluate initial predictions from a trained model

### Part 1 Generate data
To generate training dataset, use the shell file after modifying parameters such as the number of samples `Nindex_end`, noise type `noise_type` and path to save images and labels `base_path`. 
```
cd matlab_codes
bash dataset_gen.sh
```

For testset, please comment the first part in `dataset_gen.m` for training dataset, uncomment the part for test, and modify the parameters accordingly.

### Part 2 Train and infer
For training, run the `run_main.sh`, which calls `main.py` with given parameters. Explanations of parameters can be found in lines 159 to 187 in `main.py`.

For infer on test dataset, run the `run_infer.sh`, which also calls `main.py` with the argument `--train_or_test=test`. Predicted coordinates will be saved in csv file

### Part 3 Post-processing and evaluation
After getting the initial prediction results, run post-processing on them and evaluate the performance such as precision, recall, F1-score. The final predicted coordinates and performance are saved in the same folder as initial result from Part 2.
```
cd matlab_codes
bash run_postpro.sh
```


## Other info
* [ReadME template](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
