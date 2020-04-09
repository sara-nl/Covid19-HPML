import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import numpy as np


if __name__ == '__main__':
    path = 'LUNGx-CT003/03-23-2006-6667-CT NON-INFUSED CHEST-15464/5-HIGH RES-37154'
    filenames = os.listdir(path)

    # for file in filenames:
    #     path_to_file = os.path.join(path, file)
    #     image = sitk.ReadImage(path_to_file)
    #     array = sitk.GetArrayFromImage(image)
    #     print()

    ct_scan = np.zeros(shape=(512, 512, len(filenames)))

    for file in filenames:
        path_to_file = os.path.join(path, file)
        dataset = pydicom.dcmread(path_to_file)

        ct_scan[:, :, dataset.InstanceNumber-1] = dataset.pixel_array

    for i in range(len(filenames)):

        if i % 15 == 0:
            plt.imshow(ct_scan[:, :, i], cmap=plt.cm.bone)
            #plt.scatter([374], [374], c='red')
            plt.show()
            print()

