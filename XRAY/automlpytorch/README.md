# Execution
```
OMP_NUM_THREADS=40 KERAS_BACKEND="tensorflow" python autoimage.py
```

# Results
* budget : no. of trained epochs
* configs : no. of tested configs
* incumbent : best accuracy candidate
* mean : mean accuracy of candidates
* runtime : total runtime of search

## 512x512x3 | 3 classes | 280 examples / class

* Labels : ```['covid' 'no finding' 'pneumonia']```

* Overview :
```json
{
    "budget: 2": {
        "configs": 27,
        "incumbent": 0.6507936507936508,
        "mean": 0.38236560870139297
    },
    "budget: 20": {
        "configs": 3,
        "incumbent": 0.7380952380952381,
        "mean": 0.7178547378547379
    },
    "budget: 6": {
        "configs": 9,
        "incumbent": 0.7023809523809524,
        "mean": 0.6190333955773519
    },
    "budget: 60": {
        "configs": 1,
        "incumbent": 0.7738095238095237,
        "mean": 0.7738095238095237
    },
    "runtime": "0:41:32.444498"
}
```

* Incumbent configuration (60 epochs)
```json
{
    "CreateImageDataLoader:batch_size": 138,
    "ImageAugmentation:augment": true,
    "ImageAugmentation:autoaugment": false,
    "ImageAugmentation:cutout": true,
    "ImageAugmentation:cutout_holes": 3,
    "ImageAugmentation:fastautoaugment": false,
    "ImageAugmentation:length": 6,
    "LossModuleSelectorIndices:loss_module": "cross_entropy",
    "NetworkSelectorDatasetInfo:network": "resnet",
    "NetworkSelectorDatasetInfo:resnet:death_rate": 0.1361740217973929,
    "NetworkSelectorDatasetInfo:resnet:initial_filters": 22,
    "NetworkSelectorDatasetInfo:resnet:nr_main_blocks": 4,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_1": 1,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_2": 10,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_3": 6,
    "NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_4": 2,
    "NetworkSelectorDatasetInfo:resnet:res_branches_1": 2,
    "NetworkSelectorDatasetInfo:resnet:res_branches_2": 1,
    "NetworkSelectorDatasetInfo:resnet:res_branches_3": 2,
    "NetworkSelectorDatasetInfo:resnet:res_branches_4": 1,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_1": 1.7235790143369056,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_2": 1.9687665996709116,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_3": 1.1450776860841436,
    "NetworkSelectorDatasetInfo:resnet:widen_factor_4": 1.667355441782983,
    "OptimizerSelector:adamw:learning_rate": 0.00034128018103522946,
    "OptimizerSelector:adamw:weight_decay": 0.07911023857372516,
    "OptimizerSelector:optimizer": "adamw",
    "SimpleLearningrateSchedulerSelector:lr_scheduler": "step",
    "SimpleLearningrateSchedulerSelector:step:gamma": 0.03300268019066515,
    "SimpleLearningrateSchedulerSelector:step:step_size": 60,
    "SimpleTrainNode:batch_loss_computation_technique": "mixup",
    "SimpleTrainNode:mixup:alpha": 0.5545521223067762
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