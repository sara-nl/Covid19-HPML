# COVID-19 Detection Neural Network (COVNet)
This is a PyTorch implementation of the paper "[Artificial Intelligence Distinguishes COVID-19 from Community Acquired Pneumonia on Chest CT](https://pubs.rsna.org/doi/10.1148/radiol.2020200905)". It supports training, validation and testing for COVNet.

## Dataset
The COVNet paper does not publish their dataset with the paper and code. This implementation was tested on the  tested on the [SPIE-AAPM Lung CT Challenge](https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM+Lung+CT+Challenge) dataset. It contains CT images of lungs on which a classification task can be performed. 

## Model
The model proposed in this paper uses 2D convolutions on each slide of the CT scan. After propagating all the slices of the CT scan through the model, a *MaxPool* operation is performed over these features. 

## Training

Training a COVNet with default arguments. Model checkpoints and tensorboard logs are written out to a unique directory created by default within `experiments/models` and `experiments/logs` respectively after starting training.

```
python main.py
```

## Validation and Testing
You can run validation and testing on the checkpointed best model by:
```
python test.py
``` 

