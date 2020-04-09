import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import os
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from PIL import Image
from batchgenerators.transforms import noise_transforms
from batchgenerators.transforms import spatial_transforms
import pdb
import numpy as np


class SegmentationDataset(data.Dataset):

    def __init__(self, is_train=True, debug=False):
        self.is_train = is_train
        self.debug = debug

        data_path = os.path.join('data', 'tr_im.nii')
        mask_path = os.path.join('data', 'tr_mask.nii')

        imgs = nib.load(data_path)
        imgs_array = imgs.get_fdata()
        imgs_array = self.normalize_0_1(imgs_array)

        masks = nib.load(mask_path)
        masks_array = masks.get_fdata()

        if self.is_train:
            self.data = imgs_array[:, :, 0:80]
            self.masks = masks_array[:, :, 0:80]
        else:
            self.data = imgs_array[:, :, 80:]
            self.masks = masks_array[:, :, 80:]

        if self.debug:
            self.data = self.data[:, :, 0:2]
            self.masks = self.masks[:, :, 0:2]

        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])

    def __getitem__(self, item):
        """ Load an image """
        img = self.data[:, :, item]
        mask = self.masks[:, :, item]

        # Add dimension for preprocessing
        img, mask = img[None, ...], mask[None, ...]

        if self.debug:
            plt.imshow(img[0])
            plt.title('Before augmentation')
            plt.show()

        if self.is_train:
            # Default randomly mirroring the second and third axes
            img, mask = spatial_transforms.augment_mirroring(img, sample_seg=mask, axes=(0, 1))

        if self.debug:
            plt.imshow(img[0])
            plt.title('After augmentation')
            plt.show()

        # Reshape to H x W
        img, mask = img[0], mask[0]

        full_channel = np.stack([img, img, img])

        if self.is_train:
            full_channel, mask = self.do_augmentation(full_channel, mask)
        else:
            mask = mask[None, ...]

        mask = mask[0]

        if self.debug:
            plt.imshow(full_channel[0])
            plt.title('After augmentation')
            plt.show()
            plt.imshow(mask)
            plt.show()

        full_channel = torch.FloatTensor(full_channel)
        mask = torch.LongTensor(mask)

        full_channel = self.transform(full_channel)

        return full_channel, mask

    def do_augmentation(self, array, mask):
        """Augmentation for the training data.
        :array: A numpy array of size [c, x, y]
        :returns: augmented image and the corresponding mask
        """
        # normalize image to range [0, 1], then apply this transform
        patch_size = np.asarray(array.shape)[1:]
        augmented = noise_transforms.augment_gaussian_noise(array, noise_variance=(0, .015))

        # need to become [bs, c, x, y] before augment_spatial
        augmented = augmented[None, ...]
        mask = mask[None, None, ...]
        r_range = (0, (3 / 360.) * 2 * np.pi)
        cval = 0.

        augmented, mask = spatial_transforms.augment_spatial(
            augmented, seg=mask, patch_size=patch_size,
            do_elastic_deform=True, alpha=(0., 100.), sigma=(8., 13.),
            do_rotation=True, angle_x=r_range, angle_y=r_range,
            do_scale=True, scale=(.9, 1.1),
            border_mode_data='constant', border_cval_data=cval,
            order_data=3,
            p_el_per_sample=0.5,
            p_scale_per_sample=.5,
            p_rot_per_sample=.5,
            random_crop=False
        )
        mask = mask[0]
        return augmented[0], mask

    @staticmethod
    def normalize_0_1(array):
        """" Normalize all values in the array between 0 and 1 """
        return (array - array.min()) / (array.max() - array.min())

    def __len__(self):
        return self.data.shape[-1]


if __name__ == '__main__':
    seg_dataset = SegmentationDataset(is_train=True, debug=False)
    seg_dataset[0]
