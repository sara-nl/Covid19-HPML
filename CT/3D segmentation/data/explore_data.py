import nibabel as nib
import os
import numpy as np


def normalize_0_1(array):
    """" Normalize all values in the array between 0 and 1 """
    return (array - array.min()) / (array.max() - array.min())


def load_volume(data_path, mask=True):
    imgs = nib.load(data_path)
    imgs_array = imgs.get_fdata()

    if not mask:
        imgs_array = normalize_0_1(imgs_array)
    return imgs_array


if __name__ == '__main__':

    for file in os.listdir('rp_im'):
        volume_path = os.path.join('rp_im', file)
        lung_mask_path = os.path.join('rp_lung_msk', file)
        msk_path = os.path.join('rp_msk', file)

        volumes = load_volume(volume_path)
        lung_mask = load_volume(lung_mask_path, mask=True)
        mask = load_volume(msk_path, mask=True)

        print(volumes.shape)
        print(lung_mask.shape)
        print(mask.shape)
        print()
