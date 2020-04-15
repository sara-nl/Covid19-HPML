"""
Reads a text file consists of the paths of the images (one per line), then convert these images to a (compressed) HDF5 file.

Maxwell X. Cai (SURF)
"""


import numpy as np
import h5py
import os
import glob
import cv2
import argparse
import sys
import pandas as pd


def images_list_to_dataset(files_list, label, label_id, n_classes, n_channels=1):
    """
    Create a numpy ndarray dataset from a list of image paths.

    `label`: the string indicating the category name of this list, e.g., "covid";
    """
    n_images = len(files_list)

    # allocate arrays
    data = np.zeros((n_images, image_dim, image_dim, n_channels), dtype=np.uint8)

    # load the image data one by one
    for i, fullpath in enumerate(files_list):
        if not os.path.isfile(fullpath):
            continue
        fn = os.path.basename(fullpath)

        # read the image data from file
        if n_channels > 1:
            img = cv2.imread(fullpath)
        else:
            img = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_dim, image_dim))
        if n_channels == 1:
            data[i, :, :, 0] = img.astype(np.uint8)
        else:
            data[i] = img.astype(np.uint8)
        print('%s [%d/%d], file %s [%d/%d]' % (label, label_id, n_classes, fullpath, i, n_images))

    return data


n_channels = 1
image_dim = 512
images_array_dict = dict()
index_files_path = '/home/maxwellc/Covid19-HPML/XRAY/data'

index_files = ['Atelectasis_positive.txt', 'Cardiomegaly_positive.txt', 'Consolidation_positive.txt',
               'covid_positive.txt', 'Edema_positive.txt', 'Enlarged Cardiomediastinum_positive.txt',
               'Fracture_positive.txt', 'Lung Lesion_positive.txt', 'Lung Opacity_positive.txt',
               'No Finding_positive.txt', 'Pleural Effusion_positive.txt', 'Pleural Other_positive.txt',
               'Pneumonia_positive.txt', 'Pneumothorax_positive.txt', 'Support Devices_positive.txt']

class_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation',
                'covid', 'Edema', 'Enlarged_Cardiomediastinum',
                'Fracture', 'Lung_Lesion', 'Lung_Opacity',
                'No_Finding', 'Pleural_Effusion', 'Pleural_Other',
                'Pneumonia', 'Pneumothorax', 'Support_Devices']

num_classes = len(class_labels)

# Iterate the classes
for label_id, label in enumerate(class_labels):
    if os.path.isfile('%s.hdf5' % label):
        print('File %s.hdf5 already exist, skipping...' % label)
        continue

    f_index = os.path.join(index_files_path, index_files[label_id])
    files_list = pd.read_csv(f_index).values[:,0]
    print('Loading image data... ')
    data = images_list_to_dataset(files_list, label, label_id, num_classes)

    # Save to HDF5
    with h5py.File('%s.hdf5' % label, 'w') as h5f:
        print('Saving compressed datasets...', label)
        h5g = h5f.create_group(label)
        h5g.create_dataset('images', data=data, chunks=True, compression='gzip', compression_opts=3)


    # # check if the `images_array_dict` alredy have images in the same category. If so, append it; if not, assign it.
    # if label in images_array_dict.keys():
    #     images_array_dict[label] = np.append(images_array_dict[label], data, axis=0)
    # else:
    #     images_array_dict[label] = data

    # # Finally, write the datasets of all categories to the HDF5
    # for label in images_array_dict.keys():
    #     h5g = h5f.create_group(label)
    #     print('Saving compressed datasets...', label)
    #     h5g.create_dataset('images', data=data, chunks=True, compression='gzip', compression_opts=3)
    # print('Done.\n')

