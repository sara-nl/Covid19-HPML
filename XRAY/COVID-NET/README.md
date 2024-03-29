# Covid19-HPML


**Note: The COVID-Net models provided here are intended to be used as reference models that can be built upon and enhanced as new data becomes available. They are currently at a research stage and not yet intended as production-ready models (not meant for direct clinicial diagnosis), and we are working continuously to improve them as new data becomes available. Please do not use COVID-Net for self-diagnosis and seek help from your local health authorities.**


**SURFSara COVID-Net team: Valeriu Codreanu, Damian Podareanu, Ruben Hekster, Maxwell Cai, Joris Mollinga**


The world-wide pandemic response with regard to data and computer science includes so far analysing the spread of the virus, designing novel treatments or vaccines, understanding and predicting individual patient evolution as well as these implications on the healthcare system. 


As a cooperative association of the Dutch educational and research institutions, including the academic medical centers, SURF aims to support the efforts of all our members fighting against the COVID-19 pandemic. Besides offering a fast track for accessing the SURF infrastructure, we aim to offer a vision for the use of the national large-scale data, networking, compute services and expertise, in order to support researchers and to collaboratively work on this global problem. 


If there are any technical questions, please contact:
* valeriu.codreanu@surfsara.nl
* rubenh@surfsara.nl



## Requirements

The main requirements are listed below:

* Tested with Tensorflow 1.13 and 1.15
* OpenCV 4.2.0
* Python 3.6
* Numpy
* OpenCV
* Scikit-Learn
* Matplotlib

Additional requirements to generate dataset:
* PyDicom
* Pandas
* Jupyter


On **LISA** TitanRTX GPU:
```
module use /home/rubenh/environment-modules-lisa
module load 2020
module load TensorFlow/1.15.0-foss-2019b-Python-3.7.4-10.1.243
```

## COVIDx Dataset


The current COVIDx dataset is constructed by the following open source chest radiography datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
* https://github.com/hellorp1990/Covid-19-USF
 
### COVIDx data distribution


Chest radiography images distribution
|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  7966  |    8514   |    234   | 16714 |
|  test |   100  |     100   |     46   |   246 |


## Training and Evaluation
The network can take as input an image of shape (N, 512, 512, 3) and outputs the softmax probabilities as (N, 3), where N is the batch size.


### Steps for training

1. To train from scratch:
```
python -u train_keras.py \
--trainfile train_COVIDx.txt \
--testfile test_COVIDx.txt \
--data_path /nfs/managed_datasets/COVID19/XRAY/covidx_dataset_ext \
--img_size 512 \
--lr 0.00002 \
--bs 8 \
--epochs 5 \
--name covid-net-resnet512- \
--model resnet50v2
```
2. To train from an existing hdf5 file:
```
python -u train_keras.py \
--trainfile train_COVIDx.txt \
--testfile test_COVIDx.txt \
--data_path /nfs/managed_datasets/COVID19/XRAY/covidx_dataset_ext \
--img_size 512 \
--lr 0.00002 \
--bs 8 \
--epochs 5 \
--name covid-net-resnet512- \
--model resnet50v2 \
--checkpoint /home/rubenh/Covid19-HPML/checkpoint/cp-512.hdf5
```

## Results
These are the final results for COVID-Net with a ResNet50v2, EfficientNetB4 backbone with img_size (512,512,3) trained for 5 epochs.  


### COVIDNet ResNet50v2@(512)

<div class="tg-wrap" align="center"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Sensitivity (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">96.0</td>
    <td class="tg-c3ow">92.0</td>
    <td class="tg-c3ow">93.5</td>
  </tr>
</table></div>


<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Positive Predictive Value (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">93.2</td>
    <td class="tg-c3ow">93.0</td>
    <td class="tg-c3ow">97.7</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Metrics (Macro - average %)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Recall</td>
    <td class="tg-7btt">Precision</td>
    <td class="tg-7btt">F1 - Score</td>
  </tr>
  <tr>
    <td class="tg-c3ow">95.7</td>
    <td class="tg-c3ow">94.7</td>
    <td class="tg-c3ow">94.4</td>
  </tr>
</table></div>

### COVIDNet EfficientNetB4@(512)

<div class="tg-wrap" align="center"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Sensitivity (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">85.0</td>
    <td class="tg-c3ow">91.0</td>
    <td class="tg-c3ow">100.0</td>
  </tr>
</table></div>


<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Positive Predictive Value (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">94.4</td>
    <td class="tg-c3ow">85.8</td>
    <td class="tg-c3ow">92.0</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Metrics (Macro - average %)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Recall</td>
    <td class="tg-7btt">Precision</td>
    <td class="tg-7btt">F1 - Score</td>
  </tr>
  <tr>
    <td class="tg-c3ow">94.2</td>
    <td class="tg-c3ow">90.5</td>
    <td class="tg-c3ow">91.7</td>
  </tr>
</table></div>

## Confusion Matrix ResNet50v2@(512)

|         | Normal|Pneumonia |  COVID-19 |
|:-------:|:-----:|:--------:|:---------:|
|Normal   |   96  |    4     |     0     |
|Pneumonia|    7  |   92     |     1     |
|COVID-19 |    0  |    3     |    43     | 

## Confusion Matrix EfficientNetB4@(512)

|         | Normal|Pneumonia |  COVID-19 |
|:-------:|:-----:|:--------:|:---------:|
|Normal   |   85  |   15     |     0     |
|Pneumonia|    5  |   91     |     4     |
|COVID-19 |    0  |    0     |    46     | 

## Pretrained Models
|  Type | COVID-19 Sensitivity |                       Path                        |
|:-----:|:--------------------:|:-------------------------------------------------:|
|  hdf5 |         89.0         | `/home/rubenh/Covid19-HPML-static/checkpoint/cp-224-resnet50v2.hdf5`|
|  hdf5 |         93.5         | `/home/rubenh/Covid19-HPML-static/checkpoint/cp-512-resnet50v2.hdf5`|
|  hdf5 |        100.0         | `/home/rubenh/Covid19-HPML-static/checkpoint/cp-512-efficientnet.hdf5`|
