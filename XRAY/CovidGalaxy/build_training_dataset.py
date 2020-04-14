"""
Fusion the images in a list of selected categories into a single HDF5 file, to facilitate easy loading to the training pipeline.

Maxwell X. Cai (SURF)
"""


import h5py
import argparse
import glob
import os
from sklearn.preprocessing import LabelEncoder
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='dir_name', type=str, help='name of the directory containing the compressed HDF5 image files')
parser.add_argument('-e', dest='extension', type=str, default='hdf5', help='file name extension of the image file')
parser.add_argument('-n', dest='n_samples', type=int, default=1500, help='number of samples in total')
parser.add_argument('-s', '--strategy', dest='strategy', type=int, default=1, help='Sampling strategy; 1 = balanced; 2 = proportional to the original category size')
args = parser.parse_args()

if args.dir_name is None or args.dir_name == '.':
    dir_name = os.getcwd()
else:
    dir_name = args.dir_name

# If `selected_datasets` are set to an empty list, it will load all categories
# selected_datasets = []
selected_datasets = ['No_Finding.hdf5', 'Pneumonia.hdf5', 'covid.hdf5']

if len(selected_datasets) > 0:
    data_files = selected_datasets
else:
    data_files = glob.glob(os.path.join(dir_name, ('*.%s' % args.extension)))
print('Selected files', data_files, dir_name)

if len(data_files) == 0:
    print('No HDF5 data file selected. Exiting...')
    sys.exit(0)

label_lists = []
label_support = []
data_dict = dict()
support_dict = dict()
n_sampled_dict = dict()

# obtain the number of sample in each class
for f_id, data_file in enumerate(data_files):
    with h5py.File(data_file, 'r') as h5f:
        for label in h5f:
            label_support.append(h5f['%s/images' % label].shape[0])
            label_lists.append(label)
            # print('%s: %d samples' % (label, ))

# read the data according to the sampling strategy
num_classes = len(label_lists)
for f_id, data_file in enumerate(data_files):
    with h5py.File(data_file, 'r') as h5f:

        # sampling
        if args.strategy == 1:
            # balanced sampling
            n_per_class = args.n_samples // num_classes
        elif args.strategy == 2:
            # proportional sampling
            n_per_class = int(args.n_samples * (label_support[f_id] / np.sum(label_support)))

        print('Reading dataset %s and drawing %d samples...' % (data_file, n_per_class))
        label = label_lists[f_id]
        data = h5f['%s/images' % label][()]
        n_sampled_dict[label] = n_per_class
        support_dict[label] = label_support[f_id]

        random_indices = np.random.randint(label_support[f_id], size=n_per_class)
        data_dict[label] = data[random_indices]

# encode labels
le = LabelEncoder()
le.fit(label_lists)
# label_mapping = np.array([np.string_(le.classes_), le.transform(le.classes_)])

# Write to HDF5 training data file
with h5py.File('training.h5', 'w') as h5f_training:
    X = None
    Y = None
    for label in data_dict.keys():
        cid = le.transform([label])[0]
        print('%s ==> class #%d, support = %d, sampled = %d' % (label, cid, support_dict[label], n_sampled_dict[label]))

        if X is None:
            X = data_dict[label]
        else:
            X = np.append(X, data_dict[label], axis=0)

        if Y is None:
            Y = np.ones(data_dict[label].shape[0], dtype=np.int) * cid
        else:
            Y = np.append(Y, np.ones(data_dict[label].shape[0], dtype=np.int) * cid, axis=0)
    print(X.shape)

    # shuffle the data for three times
    random_indices_global = np.arange(X.shape[0])
    for t in range(3):
        np.random.shuffle(random_indices_global)

    h5f_training.create_dataset('X', data=X[random_indices_global], chunks=True, compression='gzip', compression_opts=3)
    h5f_training.create_dataset('Y', data=Y[random_indices_global], chunks=True, compression='gzip', compression_opts=3)
    # h5f_training.create_dataset('label_id', data=le.transform(le.classes_))
    # h5f_training.create_dataset('label_string', data=np.string_(le.classes_))
    # h5f_training.create_dataset('label_string', data=le.classes_, dtype=h5py.string_dtype(encoding='utf-8'))
    for label in le.classes_:
        h5f_training.attrs[label] = le.transform([label])[0]

