# LANet-DR
Lesion-aware network for diabetic retinopathy diagnosis: LANet.
scr: screening
seg: segmentation
DR: Retinopathy Diagnosis
The experiments were conducted on Ubuntu OS with AMD Ryzen 5950X Processor, 64GB RAM, and an Nvidia GeForce RTX 3090 GPU. Some modifications may need for For running the codes on Windows OS.


## train
+ Step 1, datasets.
  We involved three datasets: [IDRiD](https://idrid.grand-challenge.org/), [DDR](https://github.com/nkicsl/DDR-dataset), and [FGADR](https://csyizhou.github.io/FGADR/). 
  Download the weight named resnet50-19c8e357.pth and put it in the folder named pre-trained.
+ Step 2, preprocess the data using the code in "data_prepare".
+ Step 3, fill in the path information in "seg_config.py" and "scr_config.py".
+ Step 4, run "seg_train.py" or ```bash seg_train.sh``` to train our LANet for lesion segmentation.
+ Step 5 (not required), run "scr_train.py" or ```bash scr_train.sh``` to finetune LASNet for diabetic retinopathy classification.

## test
```bash seg_test.sh``` or ```bash scr_test.sh```.
The well-trained weights are stored in for inference without re-training, which are the same as those reported in the paper.

You can apply the model in the folder "models" to your own training/testing codes. 

```
@article{https://doi.org/10.1002/ima.22933,
author = {Xia, Xue and Zhan, Kun and Fang, Yuming and Jiang, Wenhui and Shen, Fei},
title = {Lesion-aware network for diabetic retinopathy diagnosis},
journal = {International Journal of Imaging Systems and Technology},
volume = {n/a},
number = {n/a},
pages = {},
keywords = {attention mechanism, diabetic retinopathy screening, fundus image analysis, lesion segmentation, medical image analysis, multi-task learning},
doi = {https://doi.org/10.1002/ima.22933},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/ima.22933},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/ima.22933},
}
```
