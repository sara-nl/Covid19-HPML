# Covid19-HPML


**Note: The COVID-Net models provided here are intended to be used as reference models that can be built upon and enhanced as new data becomes available. They are currently at a research stage and not yet intended as production-ready models (not meant for direct clinicial diagnosis), and we are working continuously to improve them as new data becomes available. Please do not use COVID-Net for self-diagnosis and seek help from your local health authorities.**


**SURFSara COVID-Net team: Valeriu Codreanu, Damian Podareanu, Ruben Hekster, Maxwell Cai, Joris Mollinga**


The world-wide pandemic response with regard to data and computer science includes so far analysing the spread of the virus, designing novel treatments or vaccines, understanding and predicting individual patient evolution as well as these implications on the healthcare system. 


As a cooperative association of the Dutch educational and research institutions, including the academic medical centers, SURF aims to support the efforts of all our members fighting against the COVID-19 pandemic. Besides offering a fast track for accessing the SURF infrastructure, we aim to offer a vision for the use of the national large-scale data, networking, compute services and expertise, in order to support researchers and to collaboratively work on this global problem. 


If there are any technical questions, please contact:
* valeriu.codreanu@surfsara.nl
* rubenh@surfsara.nl



## Requirements
​
The main requirements are listed below:
​
* Tested with Tensorflow 1.13 and 1.15
* OpenCV 4.2.0
* Python 3.6
* Numpy
* OpenCV
* Scikit-Learn
* Matplotlib
​
Additional requirements to generate dataset:
​
* PyDicom
* Pandas
* Jupyter
​
## COVIDx Dataset
​
​
The current COVIDx dataset is constructed by the following open source chest radiography datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge


Furthermore we extended the dataset with 204 COVID-19 positive cases: 
### COVIDx data distribution
​
Chest radiography images distribution
|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  7966  |    8514   |    234   | 16714 |
|  test |   100  |     100   |    46    |   246 |


## Training and Evaluation
The network takes as input an image of shape (N, 512, 512, 3) and outputs the softmax probabilities as (N, 3), where N is the number of batches.
If using the TF checkpoints, here are some useful tensors:
​
* input tensor: `input_1:0`
* output tensor: `dense_3/Softmax:0`
* label tensor: `dense_3_target:0`
* class weights tensor: `dense_3_sample_weights:0`
* loss tensor: `loss/mul:0`
​
### Steps for training
Releasing TF training script from pretrained model soon.
​
1. To train from scratch:
```
python -u train_keras.py \
--trainfile train_COVIDx.txt \
--testfile test_COVIDx.txt \
--data_path /nfs/managed_datasets/COVID19/XRAY/covidx_dataset_ext \
--img_size 512 \
--lr 0.00002 \
--bs 8 \
--epochs 10 \
--name covid-net-resnet512- \
--model resnet50v2
```
2. To train from an existing hdf5 file:
​
```
python -u train_keras.py \
--trainfile train_COVIDx.txt \
--testfile test_COVIDx.txt \
--data_path /nfs/managed_datasets/COVID19/XRAY/covidx_dataset_ext \
--img_size 512 \
--lr 0.00002 \
--bs 8 \
--epochs 10 \
--name covid-net-resnet512- \
--model resnet50v2 \
--checkpoint /home/$USER/Covid19-HPML/output/covid-net-resnet512-lr2e-05/cp-04-0.93.hdf5
```


## Results
These are the final results for COVID-Net with a ResNet50v2 backbone with img_size (512,512,3).   


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
​
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


## Confusion Matrix ResNet50v2@(512)

|         | Normal|Pneumonia |  COVID-19 |
|:-------:|:-----:|:--------:|:---------:|
|Normal   |   96  |    4     |     0     |
|Pneumonia|    7  |   92     |     1     |
|COVID-19 |    0  |    3     |    43     | 

## Pretrained Models
|  Type | COVID-19 Sensitivity |  Link               |
|:-----:|:--------------------:|:-------------------:|
|  hdf5 |         89.0         | [COVID-Net 224](tba)|
|  hdf5 |         93.5         | [COVID-Net 512](tba)|
