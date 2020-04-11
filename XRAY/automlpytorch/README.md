# Execution
```
OMP_NUM_THREADS=40 KERAS_BACKEND="tensorflow" python autoimage.py
```

# Results
* budget : no. of trained epochs
* configs : no. of tested configs
* incumbent : best candidate's accuracy
* mean : mean accuracy of candidates
* runtime : total runtime of search

## 512x512x3 | 2 classes | 280 examples / class

* Labels : ```['covid' 'pneumonia']```

* Overview:
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

* Incumbent configuration (60 epochs)
```json
{
    "CreateImageDataLoader:batch_size": 74,
    "ImageAugmentation:augment": true,
    "ImageAugmentation:autoaugment": true,
    "ImageAugmentation:cutout": false,
    "ImageAugmentation:fastautoaugment": false,
    "LossModuleSelectorIndices:loss_module": "cross_entropy",
    "NetworkSelectorDatasetInfo:network": "resnet",
    "NetworkSelectorDatasetInfo:resnet:death_rate": 0.015676268164241836,
    "NetworkSelectorDatasetInfo:resnet:initial_filters": 25,
    "NetworkSelectorDatasetInfo:resnet:nr_main_blocks": 3,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_1": 10,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_2": 1,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_3": 2,
    "NetworkSelectorDatasetInfo:resnet:res_branches_1": 1,
    "NetworkSelectorDatasetInfo:resnet:res_branches_2": 2,
    "NetworkSelectorDatasetInfo:resnet:res_branches_3": 5,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_1": 2.968290510365474,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_2": 0.6950674766118042,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_3": 0.5124345948516824,
    "OptimizerSelector:adamw:learning_rate": 0.00035651728792460933,
    "OptimizerSelector:adamw:weight_decay": 0.048194106757188,
    "OptimizerSelector:optimizer": "adamw",
    "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max": 1,
    "SimpleLearningrateSchedulerSelector:cosine_annealing:T_mult": 1.5323692351571623,
    "SimpleLearningrateSchedulerSelector:lr_scheduler": "cosine_annealing",
    "SimpleTrainNode:batch_loss_computation_technique": "standard"
}
```

* Hardware: ```4 x Titan RTX | 24 cores```

## 512x512x3 | 3 classes | 280 examples / class

* Labels : ```['covid' 'no finding' 'pneumonia']```

* Overview :
```json
{
    "budget: 2": {
        "configs": 270,
        "incumbent": 0.6894047619047619,
        "mean": 0.41307937796364036
    },
    "budget: 20": {
        "configs": 114,
        "incumbent": 0.8521642899584077,
        "mean": 0.6190683127011211
    },
    "budget: 6": {
        "configs": 180,
        "incumbent": 0.7982407407407408,
        "mean": 0.5589155599958713
    },
    "budget: 60": {
        "configs": 74,
        "incumbent": 0.8573809523809522,
        "mean": 0.6782328077251267
    },
    "runtime": "21:26:03.565338"
}
```

* Incumbent configuration (60 epochs)
```json
{
    "CreateImageDataLoader:batch_size": 36,
    "ImageAugmentation:augment": false,
    "ImageAugmentation:cutout": false,
    "LossModuleSelectorIndices:loss_module": "cross_entropy",
    "NetworkSelectorDatasetInfo:network": "resnet",
    "NetworkSelectorDatasetInfo:resnet:death_rate": 0.29907588933834206,
    "NetworkSelectorDatasetInfo:resnet:initial_filters": 25,
    "NetworkSelectorDatasetInfo:resnet:nr_main_blocks": 4,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_1": 4,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_2": 1,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_3": 16,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_4": 2,
    "NetworkSelectorDatasetInfo:resnet:res_branches_1": 2,
    "NetworkSelectorDatasetInfo:resnet:res_branches_2": 1,
    "NetworkSelectorDatasetInfo:resnet:res_branches_3": 3,
    "NetworkSelectorDatasetInfo:resnet:res_branches_4": 3,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_1": 1.1376708364654908,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_2": 1.0112802837600807,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_3": 1.6696154297394128,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_4": 0.6016814670740366,
    "OptimizerSelector:adamw:learning_rate": 0.0002891451382685464,
    "OptimizerSelector:adamw:weight_decay": 0.021565533178938272,
    "OptimizerSelector:optimizer": "adamw",
    "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max": 8,
    "SimpleLearningrateSchedulerSelector:cosine_annealing:T_mult": 1.4577153106471565,
    "SimpleLearningrateSchedulerSelector:lr_scheduler": "cosine_annealing",
    "SimpleTrainNode:batch_loss_computation_technique": "standard"
}
```

* Hardware : ```4 x Titan RTX | 24 cores``` 

## 512x512x3 | 8 classes | 280 examples / class

* Labels : ```['cardiomegaly' 'consolidation' 'covid' 'lung lesion' 'lung opacity' 'pleural effusion' 'pneumonia' 'pneumothorax']``` 

* Overview:
```json
{
    "budget: 2": {
        "configs": 27,
        "incumbent": 0.30894421903111746,
        "mean": 0.1622662800767377
    },
    "budget: 20": {
        "configs": 3,
        "incumbent": 0.38356529833802566,
        "mean": 0.33589658830969
    },
    "budget: 6": {
        "configs": 9,
        "incumbent": 0.3502249880801017,
        "mean": 0.27914813625277646
    },
    "budget: 60": {
        "configs": 1,
        "incumbent": 0.36960461413586415,
        "mean": 0.36960461413586415
    },
    "runtime": "1:08:25.610457"
}
```

* Incumbent configuration (60 epochs)
```json
{
    "CreateImageDataLoader:batch_size": 90,
    "ImageAugmentation:augment": false,
    "ImageAugmentation:cutout": false,
    "LossModuleSelectorIndices:loss_module": "cross_entropy",
    "NetworkSelectorDatasetInfo:network": "resnet",
    "NetworkSelectorDatasetInfo:resnet:death_rate": 0.16843232162025368,
    "NetworkSelectorDatasetInfo:resnet:initial_filters": 28,
    "NetworkSelectorDatasetInfo:resnet:nr_main_blocks": 3,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_1": 1,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_2": 6,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_3": 1,
    "NetworkSelectorDatasetInfo:resnet:res_branches_1": 2,
    "NetworkSelectorDatasetInfo:resnet:res_branches_2": 4,
    "NetworkSelectorDatasetInfo:resnet:res_branches_3": 5,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_1": 1.0603618141051467,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_2": 1.0393775912559067,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_3": 1.9337456534729118,
    "OptimizerSelector:adamw:learning_rate": 0.000518255757127549,
    "OptimizerSelector:adamw:weight_decay": 0.03654583002531868,
    "OptimizerSelector:optimizer": "adamw",
    "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max": 7,
    "SimpleLearningrateSchedulerSelector:cosine_annealing:T_mult": 1.813633692861237,
    "SimpleLearningrateSchedulerSelector:lr_scheduler": "cosine_annealing",
    "SimpleTrainNode:batch_loss_computation_technique": "standard"
}
```

* Hardware : ```4 x Titan RTX | 24 cores``` 


## 512x512x3 | 15 classes | 180 examples / class

* Labels : CheXpert + covid

* Overview:
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

* Incumbent configuration (60 epochs)
```json
{
    "CreateImageDataLoader:batch_size": 32,
    "ImageAugmentation:augment": false,
    "ImageAugmentation:cutout": false,
    "LossModuleSelectorIndices:loss_module": "cross_entropy",
    "NetworkSelectorDatasetInfo:network": "resnet",
    "NetworkSelectorDatasetInfo:resnet:death_rate": 0.22607930332669868,
    "NetworkSelectorDatasetInfo:resnet:initial_filters": 24,
    "NetworkSelectorDatasetInfo:resnet:nr_main_blocks": 2,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_1": 5,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_2": 2,
    "NetworkSelectorDatasetInfo:resnet:res_branches_1": 5,
    "NetworkSelectorDatasetInfo:resnet:res_branches_2": 5,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_1": 2.138955413671209,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_2": 1.2242994312518813,
    "OptimizerSelector:adamw:learning_rate": 0.0002916478486729365,
    "OptimizerSelector:adamw:weight_decay": 0.01300154040965886,
    "OptimizerSelector:optimizer": "adamw",
    "SimpleLearningrateSchedulerSelector:lr_scheduler": "step",
    "SimpleLearningrateSchedulerSelector:step:gamma": 0.5615610120582709,
    "SimpleLearningrateSchedulerSelector:step:step_size": 4,
    "SimpleTrainNode:batch_loss_computation_technique": "standard"
}
```

* Hardware : ```2 x Titan RTX | 40 cores```