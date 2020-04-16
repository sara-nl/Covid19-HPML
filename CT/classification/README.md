
# COVID-19 Detection Neural Network (COVNet)
This is a PyTorch implementation of GitHub Repo: **[COVID-CT](https://github.com/UCSD-AI4H/COVID-CT)**. It uses 

## Dataset
The COVID-CT-Dataset has 349 CT images containing clinical findings of COVID-19. They are located on the GitHub repo linked above. Non-COVID CT scans are also located there. A split of training-test-validation data is also provided, with the meta information (e.g., patient ID, DOI, image caption).

The images are collected from COVID19-related papers from medRxiv, bioRxiv, NEJM, JAMA, Lancet, etc. CTs containing COVID-19 abnormalities are selected by reading the figure captions in the papers. All copyrights of the data belong to the authors and publishers of these papers.

## Model
The model proposed in this GitHub is a DenseNet model. This is referred to as the baseline model. Other classification models are also possible, like the COVID-NET from [here](https://github.com/IliasPap/COVIDNet).

## Training

Training a DenseNet with default arguments. Model checkpoints and tensorboard logs are written out to a unique directory created by default within `experiments/models` and `experiments/logs` respectively after starting training.
```
python main.py
```

## Validation and Testing
You can run validation and testing on the checkpointed best model by:
```
python test.py
``` 

