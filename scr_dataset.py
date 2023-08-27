# -*- coding: utf-8 -*-

import numpy as np
from torchvision import datasets, models, transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from scr_utils import get_images

class ScrDataset(Dataset):

    def __init__(self, image_paths, mask_paths=None, label_list=None, lesions=None, transform=None):
        """
        Args:
            image_paths: paths to the original images []
            mask_paths: paths to the mask images, [[]]
            label_list: classification label list
            # class_id: id of lesions, 0:ex, 1:he, 2:ma, 3:se
            lesions: default {'EX': True, 'HE': True, 'MA': True, 'SE': True, 'BG': False}
        """
        # assert len(image_paths) == len(mask_paths)
        # assert len(image_paths) == len(label_list)
        # self.image_paths = []
        self.mask_paths = mask_paths
        # self.masks = []
        # self.images = []
        self.masks = {'EX': [], 'HE': [], 'MA': [], 'SE': []}
        self.images = []
        self.label_list = label_list
        print(np.unique(label_list, return_counts=True))
        if lesions is None:
            self.lesions = {'EX': True, 'HE': True, 'MA': True, 'SE': True, 'BG': False}
        else:
            self.lesions = lesions

        if mask_paths is not None:
            for image_path, mask_paths_ in zip(image_paths, mask_paths):
                EX_path, HE_path, MA_path, SE_path, MASK_path = mask_paths_
                self.images.append(image_path)
                self.masks['EX'].append(EX_path)
                self.masks['HE'].append(HE_path)
                self.masks['MA'].append(MA_path)
                self.masks['SE'].append(SE_path)
        else:
            self.images = image_paths

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def pil_loader(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            h, w = img.size
            return img.convert('RGB')

    def __getitem__(self, idx):
        image = self.pil_loader(self.images[idx])
        info = [image]
        label = self.label_list[idx]
        if self.mask_paths is not None:
            if self.masks['EX'][idx]:
                info.append(self.pil_loader(self.masks['EX'][idx]))
            else:
                info.append(Image.new('RGB', image.size, (0, 0, 0)))
            if self.masks['HE'][idx]:
                info.append(self.pil_loader(self.masks['HE'][idx]))
            else:
                info.append(Image.new('RGB', image.size, (0, 0, 0)))
            if self.masks['MA'][idx]:
                info.append(self.pil_loader(self.masks['MA'][idx]))
            else:
                info.append(Image.new('RGB', image.size, (0, 0, 0)))
            if self.masks['SE'][idx]:
                info.append(self.pil_loader(self.masks['SE'][idx]))
            else:
                info.append(Image.new('RGB', image.size, (0, 0, 0)))

        if self.transform:
            info = self.transform(info)
        inputs = np.array(info[0])
        if inputs.shape[2] == 3:
            inputs = np.transpose(np.array(info[0]), (2, 0, 1))
            inputs = inputs / 255.
        if self.mask_paths:
            masks = []
            mask_all = np.zeros(shape=info[0].size[::-1], dtype=np.float64)
            for j in range(1, 5):
                info[j] = np.array(info[j])
                info[j][info[j] > 127] = 255
                info[j][info[j] <= 127] = 0
                info[j] = Image.fromarray(info[j].astype('uint8'))

            if self.lesions['EX']:
                EX_mask = np.array(np.array(info[1]))[:, :, 0] / 255.0
                masks.append(EX_mask)
                mask_all = mask_all + EX_mask

            if self.lesions['HE']:
                HE_mask = np.array(np.array(info[2]))[:, :, 0] / 255.0
                masks.append(HE_mask)
                mask_all = mask_all + HE_mask

            if self.lesions['MA']:
                MA_mask = np.array(np.array(info[3]))[:, :, 0] / 255.0
                masks.append(MA_mask)
                mask_all = mask_all + MA_mask

            if self.lesions['SE']:
                SE_mask = np.array(np.array(info[4]))[:, :, 0] / 255.0
                masks.append(SE_mask)
                mask_all = mask_all + SE_mask
            if self.lesions['BG']:
                BG_mask = 1 - mask_all
                BG_mask[BG_mask<0] = 0
                masks.append(BG_mask)

            masks = np.array(masks)

            return inputs, label, masks
        else:
            return inputs, label


if __name__=="__main__":
    """
    test
    """
    import scr_config as config
    from scr_utils import get_images

    rotation_angle = config.ROTATION_ANGEL
    image_size = 512
    batchsize = config.TRAIN_BATCH_SIZE
    lesions = config.LESIONS

    image_dir = config.IMAGE_DIRS["DDR_seg"]
    label_dir = config.LABEL_DIRS["DDR_seg"]
    train_image_paths, train_label_list, train_mask_paths = get_images(image_dir, label_dir, '7', phase='train', withMasks=True)
    train_dataset = ScrDataset(train_image_paths, train_mask_paths, label_list=train_label_list, lesions=lesions, transform=None)

    train_loader = DataLoader(train_dataset, batchsize, shuffle=True)

    for inputs, label, true_masks in train_loader:
        print(inputs.shape)
        print(label)
        print(true_masks.shape)
        break

    image_dir = config.IMAGE_DIRS["DDR"]
    label_dir = config.LABEL_DIRS["DDR"]
    train_image_paths, train_label_list = get_images(image_dir, label_dir, '7', phase='train',
                                                                       withMasks=False)
    train_dataset = ScrDataset(train_image_paths, label_list=train_label_list, lesions=lesions,
                               transform=None)

    train_loader = DataLoader(train_dataset, batchsize, shuffle=True)

    for inputs, label in train_loader:
        print(inputs.shape)
        print(label)
        break

