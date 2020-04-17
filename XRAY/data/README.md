
Classes and examples
* covid positive examples 280
* Edema positive examples 2656
* No Finding positive examples 9474
* Pneumothorax positive examples 2262
* Support Devices positive examples 1424
* Enlarged Cardiomediastinum positive examples 1394
* Lung Lesion positive examples 600
* Fracture positive examples 1240
* Pleural Effusion positive examples 2695
* Atelectasis positive examples 1120
* Lung Opacity positive examples 1906
* Pneumonia positive examples 444
* Consolidation positive examples 673
* Cardiomegaly positive examples 1519
* Pleural Other positive examples 187

Data comes from:
* [COVID-Net](https://github.com/lindawangg/COVID-Net) positive COVID-19 examples from [train](https://github.com/lindawangg/COVID-Net/blob/master/train_COVIDx.txt) and [test](https://github.com/lindawangg/COVID-Net/blob/master/test_COVIDx.txt)
* [Covid-19-USF](https://github.com/hellorp1990/Covid-19-USF) positive COVID-19 examples from training and validation
* All other images come from [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)

A simple dataset exploration with a linear classifier shows:
```
                            precision    recall  f1-score   support

               atelectasis       0.00      0.00      0.00       115
              cardiomegaly       0.34      0.25      0.29       144
             consolidation       0.00      0.00      0.00        69
                     covid       0.96      0.84      0.90        31
                     edema       0.31      0.47      0.37       264
enlarged cardiomediastinum       0.00      0.00      0.00       158
                  fracture       0.33      0.01      0.02       127
               lung lesion       0.00      0.00      0.00        65
              lung opacity       0.00      0.00      0.00       179
                no finding       0.42      0.85      0.56       908
          pleural effusion       0.40      0.40      0.40       252
             pleural other       0.00      0.00      0.00        25
                 pneumonia       0.00      0.00      0.00        48
              pneumothorax       0.29      0.19      0.23       252
           support devices       0.00      0.00      0.00       152

                  accuracy                           0.40      2789
                 macro avg       0.20      0.20      0.18      2789
              weighted avg       0.27      0.40      0.30      2789

Confusion:
[[  0   3   0   0  21   0   0   0   0  75  12   0   0   4   0]
 [  0  36   0   0  28   0   0   0   0  72   7   0   0   1   0]
 [  0   1   0   0  15   0   0   0   0  26  16   0   0  11   0]
 [  0   0   0  26   2   0   0   0   0   2   0   0   0   1   0]
 [  0  16   0   0 124   0   0   0   0  99   8   0   0  17   0]
 [  0  10   0   0  12   0   2   0   0 126   2   0   0   6   0]
 [  0   2   0   0   2   0   1   0   1 117   3   0   0   1   0]
 [  0   1   0   0   2   0   0   0   0  55   3   0   0   4   0]
 [  0   4   0   0  30   0   0   0   0 121  12   0   0  12   0]
 [  0  16   0   0  59   0   0   0   0 770  37   0   0  26   0]
 [  0   6   0   0  40   0   0   0   0  91 100   0   0  15   0]
 [  0   1   0   0   4   0   0   0   0  16   0   0   0   4   0]
 [  0   4   0   1   0   0   0   0   0  38   2   0   0   3   0]
 [  0   3   0   0  43   0   0   0   0 122  36   0   0  48   0]
 [  0   3   0   0  21   0   0   0   0 108  10   0   0  10   0]]
 
Best accuracy:  39.65578842163086
```
