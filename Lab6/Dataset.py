import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import os
import cv2
import numpy as np
from PIL import Image

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            n_classes=1,
            class_num = None
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.augmentation = augmentation
        self.class_num = class_num
        self.n_classes = n_classes
    def __getitem__(self, i):
            # read data
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0)
            # image = Image.open(self.images_fps[i])
            # image = np.array(image,dtype='float32')
            # image = image.astype('float32')
            # mask = Image.open(self.masks_fps[i])
            # mask =  np.array(mask,dtype='float32')
            # mask = mask.astype('float32')
            mask_new = np.zeros(mask.shape)
            for i in range(self.n_classes):
                mask_new[mask == self.class_num[i]] = i+1
            mask =  mask_new
        
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            return image, mask.reshape(1, 320, 320)


    def __len__(self):
        return len(self.ids)
