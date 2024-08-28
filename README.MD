# LANet-DR: Lesion-aware Network for Diabetic Retinopathy Diagnosis
scr: screening
seg: segmentation
DR: retinopathy diagnosis
LANet: lesion-aware net for DR lesion segmentation
LASNet: lesion-aware disease screening net


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
The well-trained weights are available at [LANet-Baidu](https://pan.baidu.com/s/1qoLSHMcaOt7bthkRY68nqQ?pwd=fvsm) with pw ```fvsm```/[LANet-GoogleDrive](https://drive.google.com/drive/folders/1C54D5BWGO41I_X4ihGZ5sX4wPp0i13KG?usp=drive_link)/[LANet-Mega](https://mega.nz/folder/QwUAFIgR#C3H8BGYgX7BR3JTLSbJ-ow) and [LASNet-Baidu](https://pan.baidu.com/s/1uBILFc6FWQM9wRIZ2nFuEQ?pwd=1dja) with pw ```1dja```/[LASNet-GoogleDrive](https://drive.google.com/drive/folders/1tmftuBqDJDWHoa5r5vTJ0gGtRnhg8Yct?usp=drive_link)/[LASNet-Mega](https://mega.nz/folder/twlUjSRI#IeV9a0x4EzglmyPDhSxYUA) for inference.

You can apply the model in the folder "models" to your own training/testing codes. 

```
@article{https://doi.org/10.1002/ima.22933,
author = {Xia, Xue and Zhan, Kun and Fang, Yuming and Jiang, Wenhui and Shen, Fei},
title = {Lesion-aware network for diabetic retinopathy diagnosis},
journal = {International Journal of Imaging Systems and Technology},
volume = {33},
number = {6},
pages = {1914--1928},
keywords = {attention mechanism, diabetic retinopathy screening, fundus image analysis, lesion segmentation, medical image analysis, multi-task learning},
doi = {https://doi.org/10.1002/ima.22933},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/ima.22933},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/ima.22933},
}
```
[arXiv_version](https://arxiv.org/abs/2408.07264) (different with [public version](https://onlinelibrary.wiley.com/doi/abs/10.1002/ima.22933))
