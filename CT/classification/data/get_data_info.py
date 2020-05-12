import os
from glob import  glob
import imageio

def get_train_info(path):

    image_list = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        image_list.extend([os.path.join(dirpath, x) for x in filenames])

    smallest_num_pixels = float('inf')
    largest_num_pixels = float(0)

    mean_h, mean_w = [], []

    for path in image_list:
        img = imageio.imread(path)

        num_pixels = img.shape[0] * img.shape[1]

        if num_pixels < smallest_num_pixels:
            smallest_num_pixels = num_pixels
            print(f'Smallest image has shape {img.shape[0]}x{img.shape[1]}')

        if num_pixels > largest_num_pixels:
            largest_num_pixels = num_pixels
            print(f'Largest image has shape {img.shape[0]}x{img.shape[1]}')

        mean_h.append(img.shape[0])
        mean_w.append(img.shape[1])

    print(sum(mean_h) / len(mean_h))
    print(sum(mean_w) / len(mean_w))




if __name__ == '__main__':
    get_train_info('train')