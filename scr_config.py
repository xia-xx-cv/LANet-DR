# -*- coding: utf-8 -*-

LESION_IDS = {'EX':0, 'HE':1, 'MA':2, 'SE':3}

PRETRAINED = {
    "ResNet18": "",
    "ResNet34": "",
    "ResNet50": "",
    "ResNet101": "",
    "Inception_v3": "",
}

IMAGE_DIRS = {
    "DDR": "Absolute the path ro DDR dataset",
    "FGADR": "Absolute the path to FGADR dataset",
}

LABEL_DIRS = {
    "DDR": "./datas_scr/DDR",
    "FGADR": "./datas_scr/FGADR",
}

# 分割预训练模型
PRETRAINED_MODEL_PATHS = {
    "DDR": "Absolute path to the weight pretraind with LANet",
    "FGADR": "Absolute path to the weight pretraind with LANet",
}

LESIONS = {'EX': True, 'HE': True, 'MA': True, 'SE': True, 'BG': False}
CROSSENTROPY_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
ROTATION_ANGEL = 20

TRAIN_BATCH_SIZE = 8
LESION_DICE_WEIGHT = 0.
RESUME_MODEL = None
