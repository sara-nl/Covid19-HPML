# Execution
```
OMP_NUM_THREADS=40 KERAS_BACKEND="tensorflow" python autoimage.py
```

# Results
* budget : epoch budget
* configs : no. of tested configs
* incumbent : best accuracy candidate
* mean : mean accuracy of candidates
* runtime : total runtime of search

## 512x512x3 | 8 classes

Labels ```['cardiomegaly' 'consolidation' 'covid' 'lung lesion' 'lung opacity' 'pleural effusion' 'pneumonia' 'pneumothorax']```

Overview:
```json
{
    "budget: 2": {
        "configs": 27,
        "incumbent": 0.30894421903111746,
        "mean": 0.1622662800767377
    },
    "budget: 20": {
        "configs": 1,
        "incumbent": 0.35418876262626264,
        "mean": 0.35418876262626264
    },
    "budget: 6": {
        "configs": 9,
        "incumbent": 0.3502249880801017,
        "mean": 0.27914813625277646
    },
    "budget: 60": {
        "configs": 0,
        "incumbent": 0,
        "mean": 0
    },
    "runtime": "0:45:25.516459"
}
```

## 512x512x3 | 2 classes

Labels ```['covid' 'pneumonia']```

Overview:
```json
{
    "budget: 2": {
        "configs": 27,
        "incumbent": 0.9285714285714286,
        "mean": 0.5460023515579071
    },
    "budget: 20": {
        "configs": 9,
        "incumbent": 0.9722222222222222,
        "mean": 0.7358906525573192
    },
    "budget: 6": {
        "configs": 18,
        "incumbent": 0.9285714285714286,
        "mean": 0.626984126984127
    },
    "budget: 60": {
        "configs": 2,
        "incumbent": 0.9722222222222222,
        "mean": 0.9543650793650793
    },
    "runtime": "0:33:06.625056"
}
```

## 512x512x3 | 15 classes

Labels CheXpert + covid

Overview:
```json
{
    "budget: 2": {
        "configs": 92,
        "incumbent": 0.26863587540279305,
        "mean": 0.11730543444287755
    },
    "budget: 20": {
        "configs": 36,
        "incumbent": 0.33305883757011606,
        "mean": 0.19337847905830385
    },
    "budget: 6": {
        "configs": 54,
        "incumbent": 0.323838166845686,
        "mean": 0.16433643437124382
    },
    "budget: 60": {
        "configs": 24,
        "incumbent": 0.33595655806182156,
        "mean": 0.21126752427582643
    },
    "runtime": "14:39:23.872004"
}
```