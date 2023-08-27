# Lesion-aware-Network



## train

+ Step 1, download datasets [IDRiD](https://idrid.grand-challenge.org/), [DDR](https://github.com/nkicsl/DDR-dataset), and [FGADR](https://csyizhou.github.io/FGADR/). Download weights of pretrained model named resnet50-19c8e357.pth and put it in the folder named pre-trained.
+ Step 2, prepare data with the code in "data_prepare".
+ Step 3, edit "seg_config.py", fill in the paths to the datasets.
+ Step 4, run "seg_train.sh" to train LANet.
+ Step 5, edit "scr_config.py", fill in the path to the weight pretrained with LANet, and the paths to the datasets.
+ Step 6, run "scr_train.py" to finetune LASNet.





