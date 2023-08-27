# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class SegDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, lesions=None, transform=None):
        """
        Args:
            image_paths: paths to the original images []
            mask_paths: paths to the mask images, [[]]
            # class_id: id of lesions, 0:ex, 1:he, 2:ma, 3:se
            lesions: default {'EX': True, 'HE': True, 'MA': True, 'SE': True, 'BG': False}
        """
        assert len(image_paths) == len(mask_paths)
        self.masks = {'EX': [], 'HE': [], 'MA': [], 'SE': []}
        self.images = []
        if lesions is None:
            self.lesions = {'EX': True, 'HE': True, 'MA': True, 'SE': True, 'BG': False}
        else:
            self.lesions = lesions

        for image_path, mask_paths_ in zip(image_paths, mask_paths):
            EX_path, HE_path, MA_path, SE_path, MASK_path = mask_paths_
            self.images.append(image_path)
            self.masks['EX'].append(EX_path)
            self.masks['HE'].append(HE_path)
            self.masks['MA'].append(MA_path)
            self.masks['SE'].append(SE_path)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def pil_loader(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            # h, w = img.size
            return img.convert('RGB')

    def __getitem__(self, idx):
        image = self.pil_loader(self.images[idx])
        info = [image]
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
        masks = []

        if self.transform:
            info = self.transform(info)
        inputs = np.array(info[0])
        if inputs.shape[2] == 3:
            inputs = np.transpose(np.array(info[0]), (2, 0, 1))
            inputs = inputs / 255.
        mask_all = np.zeros(shape=info[0].size[::-1], dtype=np.float64)
        if len(info) > 1:
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

            return inputs, masks
        else:
            return inputs


if __name__=="__main__":
    """
    test
    """
    import seg_config as config
    from seg_utils import get_images

    rotation_angle = config.ROTATION_ANGEL
    image_size = config.IMAGE_SIZE
    image_dir = config.IMAGE_DIRS["IDRiD_seg"]
    batchsize = config.TRAIN_BATCH_SIZE
    lesions = config.LESIONS

    train_image_paths, train_mask_paths = get_images(image_dir, '7', phase='train')
    train_dataset = SegDataset(train_image_paths, train_mask_paths, lesions=lesions, transform=None)

    train_loader = DataLoader(train_dataset, batchsize, shuffle=True)

    for inputs, true_masks in train_loader:
        print(inputs.shape)
        print(true_masks.shape)
        print()
        break