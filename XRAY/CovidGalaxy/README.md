# CovidGalaxy: X-ray image recognition using deep neural network (ported from `DeepGalaxy`)

`CovidGalaxy` is an `EfficientNet`-based CNN trained for classify chest X-ray images.

## How to run the code

### Step 1: Concatenate image lists into HDF5 files
Reading many separate image files could be slow. The goal of this step is to concatenate lists of images to standalone datasets. In `images_to_hdf5.py`, the directory containing the image index files (located under the `XRAY/data` directory of this repository) is defined in the `index_file_path` variable. Modify this variable accordingly, and then run

```
python images_to_hdf5.py
```

Consequently, multiple HDF5 files will be created, each of which containing the compressed image data of the images specified in the corresponding index file. For example, `covid.hdf5` contains all images from the index file `XRAY/data/covid.txt`.

### Step 2: Build the training dataset

The training dataset can be compiled from the HDF5 files from the previous step. The script `build_training_dataset.py` defines a list of categories to fusion, then it will read the corresponding HDF5 files and sample the images from there. For example, if `selected_datasets = ['No_Finding.hdf5', 'Pneumonia.hdf5', 'covid.hdf5']`, the script will read these three HDF5 files, and then select the desired numbers of images from there, concatenate them and make them a standalone dataset.

The script `build_training_dataset.py` provides two sampling strategies. With the argument `-s 1`, it will draw a fixed number of samples from the selected datasets, which ensure that the resulting training dataset is balanced. With the argument `-s 2`, it will draw the samples proportionally. For example, suppose that dataset `A` consists of 1000 images and dataset `B` consists of 100 images, `-s 1 -n 110` will draw 55 images from dataset `A` and another 55 from dataset `B`, whereas `-s 2 -n 110` will draw 100 images from `A` and 10 images from `B`.

```
python build_training_dataset.py -n 1500 -s 1 
```

### Step 3: Carry out the training
Before starting the training, edit `self.epochs` and `self.batch_size` accordingly in `dg_train.py` to define the number of epochs and the batch size, respectively. It is also possible to use Gaussian noise to serve as a regularization approach by setting `dgtrain.use_noise = False` at the end of the file. For training a DNN on a single node with a single GPU:

```
python dg_train.py
```

For training with 2 nodes, each with 4 GPUs (please allocate the nodes first):
```
mpirun -np 8 python dg_train.py
```
