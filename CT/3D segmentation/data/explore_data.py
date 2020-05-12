import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt


def normalize_0_1(array):
    """" Normalize all values in the array between 0 and 1 """
    return (array - array.min()) / (array.max() - array.min())


def load_volume(data_path, mask=True):
    imgs = nib.load(data_path)
    imgs_array = imgs.get_fdata()

    if not mask:
        imgs_array = normalize_0_1(imgs_array)
    return imgs_array


def explore_medical_segmentation():

    for file in os.listdir('medical_segmentation/rp_im'):
        volume_path = os.path.join('medical_segmentation/rp_im', file)
        lung_mask_path = os.path.join('medical_segmentation/rp_im', file)
        msk_path = os.path.join('medical_segmentation/rp_im', file)

        volumes = load_volume(volume_path, mask=False)
        lung_masks = load_volume(lung_mask_path, mask=True)
        masks = load_volume(msk_path, mask=True)

        ct = volumes[:, :, 20]
        lung_mask = lung_masks[:, :, 20]
        mask = masks[:, :, 20]

        plt.imshow(ct)
        plt.show()

        plt.imshow(lung_mask)
        plt.show()

        plt.imshow(mask)
        plt.show()


def explore_zenodo():
    num_slices = 0
    for file in os.listdir('zenodo/ct'):
        volume_path = os.path.join('zenodo', 'ct', file)
        infection_path = os.path.join('zenodo', 'infection_mask', file)
        lung_and_infection_path = os.path.join('zenodo', 'lung_and_infection_mask', file)
        lung_path = os.path.join('zenodo', 'lung_mask', file)

        volumes = load_volume(volume_path, mask=False)
        infection_masks = load_volume(infection_path, mask=True)
        lung_and_infection_masks = load_volume((lung_and_infection_path), mask=True)
        lung_masks = load_volume(lung_path, mask=True)

        print(volumes.shape[-1])
        num_slices += volumes.shape[-1]

        for i in range(volumes.shape[-1]):
            infection_mask = infection_masks[:, :, i]

            num_infection_pixels = infection_mask.sum()

            print(f'The number of pixels with infection: {num_infection_pixels}')
    print(num_slices)

if __name__ == '__main__':

    explore_zenodo()



