# -*- coding: utf-8 -*-

LESION_IDS = {'EX':0, 'HE':1, 'MA':2, 'SE':3}
IMAGE_DIRS = {
    "IDRiD_seg": 'path to IDRiD dataset',
    "DDR_seg": "path to DDR dataset",
    "FGADR_seg": "path to FGADR dataset"
}
LESIONS = {'EX': True, 'HE': True, 'MA': True, 'SE': True, 'BG': False}

IMAGE_SIZE = 512


CROSSENTROPY_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
ROTATION_ANGEL = 20
EPOCHES = 300
TRAIN_BATCH_SIZE = 8

LESION_DICE_WEIGHT = 0. 

RESUME_MODEL = None
