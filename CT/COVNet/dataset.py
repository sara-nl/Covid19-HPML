import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from glob import glob
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from PIL import Image
from batchgenerators.transforms import noise_transforms
from batchgenerators.transforms import spatial_transforms
import pdb
import scipy


class NCovDataset(data.Dataset):
    def __init__(self, args, root_dir, stage='train'):
        super().__init__()
        self.args = args
        self.root_dir = root_dir
        self.stage = stage
        assert stage in ['train', 'val', 'test']

        if stage == 'train':
            split_file = 'training_data.xlsx'
        elif stage == 'val':
            # split_file = 'val.csv'
            raise NotImplementedError
        elif stage == 'test':
            # We just assume validation set is the same as test set
            # split_file = 'val.csv'
            raise NotImplementedError

        df = pd.read_excel(os.path.join(root_dir, split_file))
        df = df.dropna()
        df = df.drop_duplicates('Scan Number')
        df = df.set_index('Scan Number')

        self.case_ids = list(df.index)
        self.labels = df['Final Diagnosis'].values.tolist()
        self.labels = list(map(str.strip, self.labels))
        self.mapping = {label: i for i, label in enumerate(list(set(self.labels)))}

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        path_to_folder = os.path.join(self.root_dir, self.case_ids[index])
        files = []

        for r, d, f in os.walk(path_to_folder):
            for file in f:
                if '.dcm' in file:
                    files.append(os.path.join(r, file))

        ct_scan = np.zeros(shape=(512, 512, len(files)))

        for file in files:
            dataset = pydicom.dcmread(file)
            pixel_array = dataset.pixel_array
            hu = apply_modality_lut(pixel_array, dataset)
            ct_scan[:, :, dataset.InstanceNumber - 1] = hu

        ct_scan = torch.from_numpy(ct_scan).float()

        # Resize to 224 by 224 and only take 1 out of 5 elements on the Z axis into account
        ct_scan = F.interpolate(ct_scan[None, None, ...], size=(224, 224, ct_scan.shape[-1] // 5), align_corners=True,
                                mode='trilinear')
        ct_scan = ct_scan.squeeze(0).squeeze(0)
        torch.clamp(ct_scan, min=-500, max=1500, out=ct_scan)

        ct_scan = ct_scan[None, ...]
        if self.stage == 'train' and self.args.data_augment:
            # Default randomly mirroring the second and third axes
            ct_scan, _ = spatial_transforms.augment_mirroring(ct_scan, axes=(1, 2))
        ct_scan = ct_scan[0]

        ######################################################
        #  Preprocessing for both train and validation data  #
        ######################################################
        # data should be a numpy array with shape [x, y, z] or [c, x, y, z]
        full_channel = np.stack([ct_scan, ct_scan, ct_scan])
        full_channel = (full_channel - full_channel.min()) / (full_channel.max() - full_channel.min())

        if self.stage == 'train' and self.args.data_augment:
            full_channel = self.do_augmentation(full_channel)

        label = self.mapping[self.labels[index]]
        full_channel = np.transpose(full_channel, axes=(3, 0, 1, 2))
        full_channel = torch.from_numpy(full_channel).float()

        return full_channel, label, self.case_ids[index]

    def do_augmentation(self, array):
        """Augmentation for the training data.

        :array: A numpy array of size [c, x, y, z]
        :returns: augmented image and the corresponding mask

        """
        # normalize image to range [0, 1], then apply this transform
        patch_size = np.asarray(array.shape)[1:]
        augmented = noise_transforms.augment_gaussian_noise(
            array, noise_variance=(0, .015))

        # need to become [bs, c, x, y, z] before augment_spatial
        augmented = augmented[None, ...]
        mask = mask[None, None, ...]
        r_range = (0, (3 / 360.) * 2 * np.pi)
        cval = 0.

        augmented, mask = spatial_transforms.augment_spatial(
            augmented, seg=mask, patch_size=patch_size,
            do_elastic_deform=True, alpha=(0., 100.), sigma=(8., 13.),
            do_rotation=True, angle_x=r_range, angle_y=r_range, angle_z=r_range,
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

    def make_weights_for_balanced_classes(self):
        """Making sampling weights for the data samples
        :returns: sampling weigghts for dealing with class imbalance problem

        """
        n_samples = len(self.labels)
        unique, cnts = np.unique(self.labels, return_counts=True)
        cnt_dict = dict(zip(unique, cnts))

        weights = []
        for label in self.labels:
            weights.append(n_samples / float(cnt_dict[label]))
        return weights
