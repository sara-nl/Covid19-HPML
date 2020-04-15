# Segmentation
This folder contains code for running various segmentation models for COVID-19 segmented CT images. 

## Dataset
Dataset used is the dataset containing 100 slices of CT images from [here](http://medicalsegmentation.com/covid19/). Download these en move them to the *data* folder. The slices are 512 by 512 pixels and masks are included. The masks are divided into 4 classes. Since 100 slices is not a lot, we use heavy data augmentation. This includes random rotations, horizontal and vertical flips and adding Gaussian noise. The data set is split in 80% training images, and 20% validation images.

## Benchmarks
 The goal of this sub folder is to benchmark various models on this segmentation task. These models are:

 - Deeplab V3 + with ResNet 101 backbone
 - Deeplab V3 + with Xception backbone
 - Fully Convolutional Network with ResNet 50 backbone
 - Fully Convolutional Network with ResNet 101 backbone
 - Deeplab with ResNet 50 backbone
 - Deeplab with ResNet 101 backbone
 - UNet

These models are located in the *models* folder. 

## Usage
The fully convolutional models can take an input image of any dimension, as long as it has three input channels. The network takes as input an image of shape (N, 3, H, W) and outputs the logits probabilities as (N, num_classes, H, W). The path to the data is now hard coded in *dataset.py*. For running on your own data, you will have to change that. Below I give a few examples of command that can be used to rerun training to reproduce results:

To run the UNet model:

`python3 main.py --model unet` 

To run the Deeplab V3+ with Xception backbone:

`python3 main.py --model deeplabv3+ --backbone xception`

Check *options.py* for other command line options. 

## Results
In order to produce the results, all models get the same data to train on. The batch size (8) is also constant, as in the learning rate (1e-5). No learning rate scheduler was used. All models were trained for 1500 epochs. 
| Model                  | Validation mIoU | Validation accuracy |
|------------------------|-----------------|---------------------|
| UNet                   | 0.7127          | 0.9726              |
| Deeplab v3+ resnet 101 | 0.7005          | 0.9692              |
| Deeplab v3+ xception   | 0.7069          | 0.9706              |
| FCN resnet 50          | 0.7200          | 0.9718              |
| FCN resnet 101         | 0.7172          | 0.9718              |
| Deeplab resnet 50      | 0.7173          | 0.9722              |
| Deeplab resnet 101     | 0.7155          | 0.9723              |



