"""
Image input helper class for DeepGalaxy.

Maxwell X. Cai (SURF)
"""


import numpy as np
import pandas as pd
import h5py
import re


class DataIO(object):
    def __init__(self):
        self.h5f = None
        self.dset_loc = None  # h5 path of the dataset (for partial loading)
        self.local_index = None  # index of an element within a h5 dataset
        self.flattened_labels = None

    def load_all(self, h5fn):
        """
        Load all data that match the dataset name pattern.
        """
        if self.h5f is None:
            self.h5f = h5py.File(h5fn, 'r')
        X = self.load_dataset('X')
        Y = self.load_dataset('Y')

        return X, Y

    def load_partial(self, h5fn, hvd_size=None, hvd_rank=None):
        """
        Load the data that match the dataset name pattern partially according to the rank.
        """
        if self.h5f is None:
            self.h5f = h5py.File(h5fn, 'r')

        if hvd_size is None or hvd_rank is None:
            return self.load_all(h5fn)

        n_images_total = self.h5f['Y'].shape[0]

        n_images_per_proc = int(n_images_total / hvd_size)
        print('hvd_size = %d, n_images_per_proc = %d, total_images = %d' % (hvd_size, n_images_per_proc, n_images_total))
        idx_global_start = n_images_per_proc * hvd_rank
        idx_global_end = n_images_per_proc * hvd_rank + n_images_per_proc
        print('Rank %d gets %d images, from #%d to #%d' % (hvd_rank, n_images_per_proc, idx_global_start, idx_global_end))

        X = self.h5f['X'][idx_global_start:idx_global_end]
        Y = self.h5f['Y'][idx_global_start:idx_global_end]
        print('X shape', X.shape, X[0].shape)
        return X, Y

    def load_dataset(self, h5_path):
        """
        Load a dataset according to the h5_path.
        """
        images = self.h5f[h5_path][()]
        if images.dtype == np.uint8:
            # if uint8, then the range is [0, 255]. Normalize to [0, 1]
            # if float32, don't do anything since the range is already [0, 1]
            images = images.astype(np.float) / 255
        elif images.shape[-1] == 1:
            # just gray scale float images. Repeat along the channel
            # old_shape = images.shape
            # images = np.repeat(images.astype(np.float32), 3, axis=(len(old_shape)-1))
            # print('Repeating...', old_shape, images.shape)
            pass
        return images

    def flatten_index(self, h5fn, dset_name_pattern, camera_pos=0, t_lim=None):
        """
        Traverse the HDF5 dataset tree, and flatten the structure (not the data, just the structure)
        so that it is more easily diviable accroding to the `hvd_size()`.
        """
        dset_loc = []
        local_index = []
        label_flattened = []
        if self.h5f is None:
            self.h5f = h5py.File(h5fn, 'r')
        full_dset_list = self.h5f.keys()
        r = re.compile(dset_name_pattern)
        matched_dset_list = list(filter(r.match, full_dset_list))
        print('Flattening... Selected datasets: %s' % matched_dset_list)
        images_set = []
        labels_cpos_set = []
        if isinstance(camera_pos, int):
            camera_pos = [camera_pos]  # convert to list
        elif isinstance(camera_pos, str):
            if camera_pos == '*':
                camera_pos = range(0, 14)
        print('Selected camera positions: %s' % camera_pos)
        for dset_name in matched_dset_list:
            for cpos in camera_pos:
                dset_path_full = '/%s/images_camera_%02d' % (dset_name, cpos)
                label_dset_path_full = '/%s/t_myr_camera_%02d' % (dset_name, cpos)
                print('Obtaining the shape of dataset %s' % dset_path_full)
                shape = self.h5f[dset_path_full].shape
                labels = self.h5f[label_dset_path_full][()]

                labels_t_ = (labels / 5).astype(np.int)
                labels_t_ = labels_t_ - np.min(labels_t_)

                dset_loc = np.append(dset_loc, [dset_path_full]*shape[0])
                local_index = np.append(local_index, np.arange(0, shape[0], dtype=np.int))
                label_flattened = np.append(label_flattened, labels_t_)
        return dset_loc, local_index.astype(np.int), label_flattened


    def get_labels(self, dset_name, camera_pos=0):
        print('Getting labels...')
        size_ratios = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5])
        mass_ratios = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5])
        s = float(dset_name.split('_')[1])
        m = float(dset_name.split('_')[3])
        cat_t = self.h5f['%s/t_myr_camera_%02d' % (dset_name, camera_pos)].value
        cat_s = np.argwhere(size_ratios==s)[0, 0] * np.ones(cat_t.shape, dtype=np.int)
        cat_m = np.argwhere(mass_ratios==m)[0, 0] * np.ones(cat_t.shape, dtype=np.int)
        return cat_m, cat_s, cat_t

