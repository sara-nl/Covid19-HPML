"""
Convert the CT images to 2D slices which can be used for 2D segmentation / classification algorithms
"""
import os
import nibabel as nib
import numpy as np
import imageio
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


def explore_zenodo():

    j = 0

    for file in os.listdir('zenodo/ct'):
        volume_path = os.path.join('zenodo', 'ct', file)
        infection_path = os.path.join('zenodo', 'infection_mask', file)
        lung_and_infection_path = os.path.join('zenodo', 'lung_and_infection_mask', file)
        lung_path = os.path.join('zenodo', 'lung_mask', file)

        volumes = load_volume(volume_path, mask=False)
        infection_masks = load_volume(infection_path, mask=True)

        for i in range(volumes.shape[-1]):

            ct = (volumes[:, :, i] * 255).astype(np.uint8)

            infection_mask = infection_masks[:, :, i]
            num_infection_pixels = infection_mask.sum()

            if num_infection_pixels > 0:
                label = 'covid'
            else:
                label = 'noncovid'

            os.makedirs(label, exist_ok=True)
            imageio.imsave(f'{label}/{j}.png', ct)

            j += 1


if __name__ == '__main__':
    explore_zenodo()



