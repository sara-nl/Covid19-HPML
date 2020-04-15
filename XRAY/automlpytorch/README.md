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

## 256x256x3 | 3 classes | 280 examples / class

* Labels : ```['covid' 'no finding' 'pneumonia']```

* Overview :
```json
{
    "budget: 2": {
        "configs": 20,
        "incumbent": 0.6549242424242424,
        "mean": 0.4642412246318498
    },
    "budget: 20": {
        "configs": 12,
        "incumbent": 0.7359126984126985,
        "mean": 0.6609446649029982
    },
    "budget: 6": {
        "configs": 14,
        "incumbent": 0.675,
        "mean": 0.6073819847480563
    },
    "budget: 60": {
        "configs": 5,
        "incumbent": 0.7791666666666667,
        "mean": 0.6748898982281336
    },
    "runtime": "4:22:56.664665"
}
```

* Incumbent configuration (60 epochs) [1]
```json
[[1, 0, 8], 60.0, {"submitted": 1586639778.370423, "started": 1586639778.3705494, "finished": 1586640966.0534487}, {"loss": -0.7791666666666667, "info": {"train_loss": 0.7766528717229064, "train_balanced_accuracy": 0.6164189321010349, "val_balanced_accuracy": 0.7791666666666667, "epochs": 37, "model_parameters": 7706577, "learning_rate": 7.530287316575816e-10, "checkpoint": "3class_280balanced_ba_ce/checkpoints/checkpoint_(1, 0, 8)_Budget_60.pt", "train_datapoints": 717, "val_datapoints": 80}, "losses": -0.7791666666666667}, null]
```

* Incumbent configuration (60 epochs) [2]
```json
{
    "CreateImageDataLoader:batch_size": 10,
    "ImageAugmentation:augment": false,
    "ImageAugmentation:cutout": true,
    "ImageAugmentation:cutout_holes": 2,
    "ImageAugmentation:length": 12,
    "LossModuleSelectorIndices:loss_module": "cross_entropy",
    "NetworkSelectorDatasetInfo:darts:auxiliary": false,
    "NetworkSelectorDatasetInfo:darts:drop_path_prob": 0.1,
    "NetworkSelectorDatasetInfo:darts:init_channels": 36,
    "NetworkSelectorDatasetInfo:darts:layers": 20,
    "NetworkSelectorDatasetInfo:densenet:blocks": 4,
    "NetworkSelectorDatasetInfo:densenet:growth_rate": 33,
    "NetworkSelectorDatasetInfo:densenet:layer_in_block_1": 3,
    "NetworkSelectorDatasetInfo:densenet:layer_in_block_2": 10,
    "NetworkSelectorDatasetInfo:densenet:layer_in_block_3": 30,
    "NetworkSelectorDatasetInfo:densenet:layer_in_block_4": 14,
    "NetworkSelectorDatasetInfo:densenet:use_dropout": false,
    "NetworkSelectorDatasetInfo:network": "densenet",
    "OptimizerSelector:adam:learning_rate": 0.00016948104783605334,
    "OptimizerSelector:adam:weight_decay": 0.06784769305666362,
    "OptimizerSelector:optimizer": "adam",
    "SimpleLearningrateSchedulerSelector:lr_scheduler": "step",
    "SimpleLearningrateSchedulerSelector:step:gamma": 0.0021078766999122105,
    "SimpleLearningrateSchedulerSelector:step:step_size": 13,
    "SimpleTrainNode:batch_loss_computation_technique": "mixup",
    "SimpleTrainNode:mixup:alpha": 0.2962837024692073
}
```

* Hardware: ```4 x Titan RTX | 24 cores```

## 32x32x3 | 2 classes | 280 examples / class

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

## 32x32x3 | 3 classes | 280 examples / class

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

## 32x32x3 | 8 classes | 280 examples / class

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


## 32x32x3 | 15 classes | 180 examples / class

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

# Breakdown of one run
This is the result for

```Python
autoPyTorch = AutoNetImageClassification(min_workers=4, \
                                        log_level='info', \
                                        budget_type='epochs', \
                                        min_budget=1, \
                                        max_budget=60, \
                                        num_iterations=1, \
                                        cuda=True)
autoPyTorch.fit(images, labels, use_tensorboard_logger=True, validation_split=0.1)
```

## Autonet config options
```Python
autoPyTorch.get_autonet_config_file_parser().print_help()
```
```
name                                default                           choices                  type                                     
========================================================================================================================================
additional_logs                     []                                []                       <class 'str'>                            
----------------------------------------------------------------------------------------------------------------------------------------
additional_metrics                  []                                [accuracy,               <class 'str'>                            
                                                                       auc_metric,                                                      
                                                                       pac_metric,                                                      
                                                                       balanced_accuracy,                                               
                                                                       cross_entropy]                                                   
----------------------------------------------------------------------------------------------------------------------------------------
algorithm                           bohb                              [bohb,                   <class 'str'>                            
                                                                       hyperband]                                                       
----------------------------------------------------------------------------------------------------------------------------------------
batch_loss_computation_techniques   [standard,                        [standard,               <class 'str'>                            
                                     mixup]                            mixup]                                                           
----------------------------------------------------------------------------------------------------------------------------------------
budget_type                         time                              [time,                   <class 'str'>                            
                                                                       epochs]                                                          
----------------------------------------------------------------------------------------------------------------------------------------
cuda                                True                              [True,                   <function to_bool at 0x2b0e481ca730>     
                                                                       False]                                                           
----------------------------------------------------------------------------------------------------------------------------------------
cv_splits                           1                                 None                     <class 'int'>                            
----------------------------------------------------------------------------------------------------------------------------------------
dataloader_cache_size_mb            0                                 None                     <class 'int'>                            
----------------------------------------------------------------------------------------------------------------------------------------
dataloader_worker                   1                                 None                     <class 'int'>                            
----------------------------------------------------------------------------------------------------------------------------------------
dataset_order                       None                              None                     <class 'int'>                            
	info: Only used for multiple datasets.
----------------------------------------------------------------------------------------------------------------------------------------
default_dataset_download_dir        /home/damian/COVID/Auto-PyTorch   None                     directory                                
	info: Directory default datasets will be downloaded to.
----------------------------------------------------------------------------------------------------------------------------------------
eta                                 3                                 None                     <class 'float'>                          
	info: eta parameter of Hyperband.
----------------------------------------------------------------------------------------------------------------------------------------
evaluate_on_train_data              True                              None                     <function to_bool at 0x2b0e481ca730>     
----------------------------------------------------------------------------------------------------------------------------------------
file_extensions                     [.png,                            None                     <class 'str'>                            
                                     .jpg,                                                                                              
                                     .JPEG,                                                                                             
                                     .pgm]                                                                                              
----------------------------------------------------------------------------------------------------------------------------------------
final_activation                    softmax                           [softmax]                <class 'str'>                            
----------------------------------------------------------------------------------------------------------------------------------------
global_results_dir                  None                              None                     directory                                
----------------------------------------------------------------------------------------------------------------------------------------
half_num_cv_splits_below_budget     0                                 None                     <class 'float'>                          
----------------------------------------------------------------------------------------------------------------------------------------
hyperparameter_search_space_updates None                              None                     [directory,                              
                                                                                                <function parse_hyperparameter_search_] 
	info: object of type HyperparameterSearchSpaceUpdates
----------------------------------------------------------------------------------------------------------------------------------------
images_root_folders                 [/home/damian/COVID/Auto-PyTorch] None                     directory                                
	info: Directory relative to which image paths are given.
----------------------------------------------------------------------------------------------------------------------------------------
images_shape                        [3,                               None                     <class 'int'>                            
                                     32,                                                                                                
                                     32]                                                                                                
	info: Image size input to the networks, images will be rescaled to this.
----------------------------------------------------------------------------------------------------------------------------------------
increase_number_of_trained_datasets False                             None                     <function to_bool at 0x2b0e481ca730>     
	info: Only used for multiple datasets.
----------------------------------------------------------------------------------------------------------------------------------------
keep_only_incumbent_checkpoints     True                              None                     <function to_bool at 0x2b0e481ca730>     
----------------------------------------------------------------------------------------------------------------------------------------
log_level                           warning                           [debug,                  <class 'str'>                            
                                                                       info,                                                            
                                                                       warning,                                                         
                                                                       error,                                                           
                                                                       critical]                                                        
----------------------------------------------------------------------------------------------------------------------------------------
loss_modules                        [cross_entropy,                   [cross_entropy,          <class 'str'>                            
                                     cross_entropy_weighted]           cross_entropy_weighted]                                          
----------------------------------------------------------------------------------------------------------------------------------------
lr_scheduler                        [cosine_annealing,                [cosine_annealing,       <class 'str'>                            
                                     cyclic,                           cyclic,                                                          
                                     step,                             step,                                                            
                                     adapt,                            adapt,                                                           
                                     plateau,                          plateau,                                                         
                                     alternating_cosine,               alternating_cosine,                                              
                                     exponential,                      exponential,                                                     
                                     none]                             none]                                                            
----------------------------------------------------------------------------------------------------------------------------------------
max_budget                          6000                              None                     <class 'float'>                          
	info: Max budget for fitting configurations.
----------------------------------------------------------------------------------------------------------------------------------------
max_class_size                      None                              None                     <class 'int'>                            
----------------------------------------------------------------------------------------------------------------------------------------
max_runtime                         24000                             None                     <class 'float'>                          
	info: Total time for the run.
----------------------------------------------------------------------------------------------------------------------------------------
memory_limit_mb                     1000000                           None                     <class 'int'>                            
----------------------------------------------------------------------------------------------------------------------------------------
min_budget                          120                               None                     <class 'float'>                          
	info: Min budget for fitting configurations.
----------------------------------------------------------------------------------------------------------------------------------------
min_budget_for_cv                   0                                 None                     <class 'float'>                          
----------------------------------------------------------------------------------------------------------------------------------------
min_workers                         1                                 None                     <class 'int'>                            
----------------------------------------------------------------------------------------------------------------------------------------
minimize                            False                             [True,                   <function to_bool at 0x2b0e481ca730>     
                                                                       False]                                                           
----------------------------------------------------------------------------------------------------------------------------------------
network_interface_name              admin                             None                     <class 'str'>                            
----------------------------------------------------------------------------------------------------------------------------------------
networks                            [densenet,                        [densenet,               <class 'str'>                            
                                     densenet_flexible,                densenet_flexible,                                               
                                     resnet,                           resnet,                                                          
                                     resnet152,                        resnet152,                                                       
                                     darts,                            darts,                                                           
                                     mobilenet]                        mobilenet]                                                       
----------------------------------------------------------------------------------------------------------------------------------------
num_iterations                      inf                               None                     <class 'float'>                          
	info: Number of successive halving iterations
----------------------------------------------------------------------------------------------------------------------------------------
optimize_metric                     accuracy                          [accuracy,               <class 'str'>                            
                                                                       auc_metric,                                                      
                                                                       pac_metric,                                                      
                                                                       balanced_accuracy,                                               
                                                                       cross_entropy]                                                   
	info: This is the meta train metric BOHB will try to optimize.
----------------------------------------------------------------------------------------------------------------------------------------
optimizer                           [adam,                            [adam,                   <class 'str'>                            
                                     adamw,                            adamw,                                                           
                                     sgd,                              sgd,                                                             
                                     rmsprop]                          rmsprop]                                                         
----------------------------------------------------------------------------------------------------------------------------------------
random_seed                         991183599                         None                     <class 'int'>                            
	info: Make sure to specify the same seed for all workers.
----------------------------------------------------------------------------------------------------------------------------------------
result_logger_dir                   .                                 None                     directory                                
----------------------------------------------------------------------------------------------------------------------------------------
run_id                              0                                 None                     <class 'str'>                            
	info: Unique id for each run.
----------------------------------------------------------------------------------------------------------------------------------------
save_checkpoints                    False                             [True,                   <function to_bool at 0x2b0e481ca730>     
                                                                       False]                                                           
	info: Wether to save state dicts as checkpoints.
----------------------------------------------------------------------------------------------------------------------------------------
shuffle                             True                              [True,                   <function to_bool at 0x2b0e481ca730>     
                                                                       False]                                                           
----------------------------------------------------------------------------------------------------------------------------------------
task_id                             -1                                None                     <class 'int'>                            
	info: ID for each worker, if you run AutoNet on a cluster. Set to -1, if you run it locally. 
----------------------------------------------------------------------------------------------------------------------------------------
tensorboard_images_count            0                                 None                     <class 'int'>                            
----------------------------------------------------------------------------------------------------------------------------------------
tensorboard_min_log_interval        30                                None                     <class 'int'>                            
----------------------------------------------------------------------------------------------------------------------------------------
use_stratified_cv_split             True                              [True,                   <function to_bool at 0x2b0e481ca730>     
                                                                       False]                                                           
----------------------------------------------------------------------------------------------------------------------------------------
use_tensorboard_logger              False                             None                     <function to_bool at 0x2b0e481ca730>     
----------------------------------------------------------------------------------------------------------------------------------------
validation_split                    0.0                               [0,                      <class 'float'>                          
                                                                       1]                                                               
----------------------------------------------------------------------------------------------------------------------------------------
working_dir                         .                                 None                     directory                                
----------------------------------------------------------------------------------------------------------------------------------------
```

## Autonet Search space configuration
```Python
autoPyTorch.get_hyperparameter_search_space()
```
```
Configuration space object:
  Hyperparameters:
    CreateImageDataLoader:batch_size, Type: UniformInteger, Range: [32, 160], Default: 72, on log-scale
    ImageAugmentation:augment, Type: Categorical, Choices: {True, False}, Default: True
    ImageAugmentation:autoaugment, Type: Categorical, Choices: {True, False}, Default: True
    ImageAugmentation:cutout, Type: Categorical, Choices: {True, False}, Default: True
    ImageAugmentation:cutout_holes, Type: UniformInteger, Range: [1, 3], Default: 2
    ImageAugmentation:fastautoaugment, Type: Categorical, Choices: {True, False}, Default: True
    ImageAugmentation:length, Type: UniformInteger, Range: [0, 20], Default: 10
    LossModuleSelectorIndices:loss_module, Type: Categorical, Choices: {cross_entropy_weighted, cross_entropy}, Default: cross_entropy_weighted
    NetworkSelectorDatasetInfo:mobilenet:initial_filters, Type: UniformInteger, Range: [8, 32], Default: 16, on log-scale
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_1, Type: Categorical, Choices: {3, 5}, Default: 3
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_2, Type: Categorical, Choices: {3, 5}, Default: 3
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_3, Type: Categorical, Choices: {3, 5}, Default: 3
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_4, Type: Categorical, Choices: {3, 5}, Default: 3
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_5, Type: Categorical, Choices: {3, 5}, Default: 3
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_6, Type: Categorical, Choices: {3, 5}, Default: 3
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_7, Type: Categorical, Choices: {3, 5}, Default: 3
    NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks, Type: UniformInteger, Range: [3, 7], Default: 5
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_1, Type: UniformInteger, Range: [1, 4], Default: 2
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_2, Type: UniformInteger, Range: [1, 4], Default: 2
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_3, Type: UniformInteger, Range: [1, 4], Default: 2
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_4, Type: UniformInteger, Range: [1, 4], Default: 2
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_5, Type: UniformInteger, Range: [1, 4], Default: 2
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_6, Type: UniformInteger, Range: [1, 4], Default: 2
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_7, Type: UniformInteger, Range: [1, 4], Default: 2
    NetworkSelectorDatasetInfo:mobilenet:op_type_1, Type: Categorical, Choices: {inverted_residual, dwise_sep_conv}, Default: inverted_residual
    NetworkSelectorDatasetInfo:mobilenet:op_type_2, Type: Categorical, Choices: {inverted_residual, dwise_sep_conv}, Default: inverted_residual
    NetworkSelectorDatasetInfo:mobilenet:op_type_3, Type: Categorical, Choices: {inverted_residual, dwise_sep_conv}, Default: inverted_residual
    NetworkSelectorDatasetInfo:mobilenet:op_type_4, Type: Categorical, Choices: {inverted_residual, dwise_sep_conv}, Default: inverted_residual
    NetworkSelectorDatasetInfo:mobilenet:op_type_5, Type: Categorical, Choices: {inverted_residual, dwise_sep_conv}, Default: inverted_residual
    NetworkSelectorDatasetInfo:mobilenet:op_type_6, Type: Categorical, Choices: {inverted_residual, dwise_sep_conv}, Default: inverted_residual
    NetworkSelectorDatasetInfo:mobilenet:op_type_7, Type: Categorical, Choices: {inverted_residual, dwise_sep_conv}, Default: inverted_residual
    NetworkSelectorDatasetInfo:mobilenet:out_filters_1, Type: Categorical, Choices: {12, 16, 20}, Default: 12
    NetworkSelectorDatasetInfo:mobilenet:out_filters_2, Type: Categorical, Choices: {18, 24, 30}, Default: 18
    NetworkSelectorDatasetInfo:mobilenet:out_filters_3, Type: Categorical, Choices: {24, 32, 40}, Default: 24
    NetworkSelectorDatasetInfo:mobilenet:out_filters_4, Type: Categorical, Choices: {48, 64, 80}, Default: 48
    NetworkSelectorDatasetInfo:mobilenet:out_filters_5, Type: Categorical, Choices: {72, 96, 120}, Default: 72
    NetworkSelectorDatasetInfo:mobilenet:out_filters_6, Type: Categorical, Choices: {120, 160, 200}, Default: 120
    NetworkSelectorDatasetInfo:mobilenet:out_filters_7, Type: Categorical, Choices: {240, 320, 400}, Default: 240
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_1, Type: Categorical, Choices: {0, 0.25}, Default: 0
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_2, Type: Categorical, Choices: {0, 0.25}, Default: 0
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_3, Type: Categorical, Choices: {0, 0.25}, Default: 0
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_4, Type: Categorical, Choices: {0, 0.25}, Default: 0
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_5, Type: Categorical, Choices: {0, 0.25}, Default: 0
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_6, Type: Categorical, Choices: {0, 0.25}, Default: 0
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_7, Type: Categorical, Choices: {0, 0.25}, Default: 0
    NetworkSelectorDatasetInfo:mobilenet:skip_con_1, Type: Categorical, Choices: {True, False}, Default: True
    NetworkSelectorDatasetInfo:mobilenet:skip_con_2, Type: Categorical, Choices: {True, False}, Default: True
    NetworkSelectorDatasetInfo:mobilenet:skip_con_3, Type: Categorical, Choices: {True, False}, Default: True
    NetworkSelectorDatasetInfo:mobilenet:skip_con_4, Type: Categorical, Choices: {True, False}, Default: True
    NetworkSelectorDatasetInfo:mobilenet:skip_con_5, Type: Categorical, Choices: {True, False}, Default: True
    NetworkSelectorDatasetInfo:mobilenet:skip_con_6, Type: Categorical, Choices: {True, False}, Default: True
    NetworkSelectorDatasetInfo:mobilenet:skip_con_7, Type: Categorical, Choices: {True, False}, Default: True
    NetworkSelectorDatasetInfo:mobilenet:stride_1, Type: Categorical, Choices: {1, 2}, Default: 1
    NetworkSelectorDatasetInfo:mobilenet:stride_2, Type: Categorical, Choices: {1, 2}, Default: 1
    NetworkSelectorDatasetInfo:mobilenet:stride_3, Type: Categorical, Choices: {1, 2}, Default: 1
    NetworkSelectorDatasetInfo:mobilenet:stride_4, Type: Categorical, Choices: {1, 2}, Default: 1
    NetworkSelectorDatasetInfo:mobilenet:stride_5, Type: Categorical, Choices: {1, 2}, Default: 1
    NetworkSelectorDatasetInfo:mobilenet:stride_6, Type: Categorical, Choices: {1, 2}, Default: 1
    NetworkSelectorDatasetInfo:mobilenet:stride_7, Type: Categorical, Choices: {1, 2}, Default: 1
    NetworkSelectorDatasetInfo:network, Type: Categorical, Choices: {resnet, mobilenet}, Default: resnet
    NetworkSelectorDatasetInfo:resnet:death_rate, Type: UniformFloat, Range: [0.0, 1.0], Default: 0.5
    NetworkSelectorDatasetInfo:resnet:initial_filters, Type: UniformInteger, Range: [8, 32], Default: 16, on log-scale
    NetworkSelectorDatasetInfo:resnet:nr_main_blocks, Type: UniformInteger, Range: [2, 4], Default: 3
    NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_1, Type: UniformInteger, Range: [1, 16], Default: 4, on log-scale
    NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_2, Type: UniformInteger, Range: [1, 16], Default: 4, on log-scale
    NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_3, Type: UniformInteger, Range: [1, 16], Default: 4, on log-scale
    NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_4, Type: UniformInteger, Range: [1, 16], Default: 4, on log-scale
    NetworkSelectorDatasetInfo:resnet:res_branches_1, Type: UniformInteger, Range: [1, 5], Default: 3
    NetworkSelectorDatasetInfo:resnet:res_branches_2, Type: UniformInteger, Range: [1, 5], Default: 3
    NetworkSelectorDatasetInfo:resnet:res_branches_3, Type: UniformInteger, Range: [1, 5], Default: 3
    NetworkSelectorDatasetInfo:resnet:res_branches_4, Type: UniformInteger, Range: [1, 5], Default: 3
    NetworkSelectorDatasetInfo:resnet:widen_factor_1, Type: UniformFloat, Range: [0.5, 4.0], Default: 1.4142135624, on log-scale
    NetworkSelectorDatasetInfo:resnet:widen_factor_2, Type: UniformFloat, Range: [0.5, 4.0], Default: 1.4142135624, on log-scale
    NetworkSelectorDatasetInfo:resnet:widen_factor_3, Type: UniformFloat, Range: [0.5, 4.0], Default: 1.4142135624, on log-scale
    NetworkSelectorDatasetInfo:resnet:widen_factor_4, Type: UniformFloat, Range: [0.5, 4.0], Default: 1.4142135624, on log-scale
    OptimizerSelector:adamw:learning_rate, Type: UniformFloat, Range: [0.0001, 0.1], Default: 0.0031622777, on log-scale
    OptimizerSelector:adamw:weight_decay, Type: UniformFloat, Range: [1e-05, 0.1], Default: 0.050005
    OptimizerSelector:optimizer, Type: Categorical, Choices: {sgd, adamw}, Default: sgd
    OptimizerSelector:sgd:learning_rate, Type: UniformFloat, Range: [0.0001, 0.1], Default: 0.0031622777, on log-scale
    OptimizerSelector:sgd:momentum, Type: UniformFloat, Range: [0.1, 0.99], Default: 0.3146426545, on log-scale
    OptimizerSelector:sgd:weight_decay, Type: UniformFloat, Range: [1e-05, 0.1], Default: 0.050005
    SimpleLearningrateSchedulerSelector:cosine_annealing:T_max, Type: UniformInteger, Range: [1, 100], Default: 10, on log-scale
    SimpleLearningrateSchedulerSelector:cosine_annealing:T_mult, Type: UniformFloat, Range: [1.0, 2.0], Default: 1.5
    SimpleLearningrateSchedulerSelector:lr_scheduler, Type: Categorical, Choices: {step, cosine_annealing}, Default: step
    SimpleLearningrateSchedulerSelector:step:gamma, Type: UniformFloat, Range: [0.001, 0.99], Default: 0.0314642654, on log-scale
    SimpleLearningrateSchedulerSelector:step:step_size, Type: UniformInteger, Range: [1, 100], Default: 10, on log-scale
    SimpleTrainNode:batch_loss_computation_technique, Type: Categorical, Choices: {standard, mixup}, Default: standard
    SimpleTrainNode:mixup:alpha, Type: UniformFloat, Range: [0.0, 1.0], Default: 1.0
  Conditions:
    ImageAugmentation:autoaugment | ImageAugmentation:augment == True
    ImageAugmentation:cutout_holes | ImageAugmentation:cutout == True
    ImageAugmentation:fastautoaugment | ImageAugmentation:augment == True
    ImageAugmentation:length | ImageAugmentation:cutout == True
    NetworkSelectorDatasetInfo:mobilenet:initial_filters | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_1 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_2 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_3 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_4 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 3
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_5 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 4
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_6 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 5
    NetworkSelectorDatasetInfo:mobilenet:kernel_size_7 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 6
    NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_1 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_2 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_3 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_4 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 3
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_5 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 4
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_6 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 5
    NetworkSelectorDatasetInfo:mobilenet:nr_sub_blocks_7 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 6
    NetworkSelectorDatasetInfo:mobilenet:op_type_1 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:op_type_2 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:op_type_3 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:op_type_4 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 3
    NetworkSelectorDatasetInfo:mobilenet:op_type_5 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 4
    NetworkSelectorDatasetInfo:mobilenet:op_type_6 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 5
    NetworkSelectorDatasetInfo:mobilenet:op_type_7 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 6
    NetworkSelectorDatasetInfo:mobilenet:out_filters_1 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:out_filters_2 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:out_filters_3 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:out_filters_4 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 3
    NetworkSelectorDatasetInfo:mobilenet:out_filters_5 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 4
    NetworkSelectorDatasetInfo:mobilenet:out_filters_6 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 5
    NetworkSelectorDatasetInfo:mobilenet:out_filters_7 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 6
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_1 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_2 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_3 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_4 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 3
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_5 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 4
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_6 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 5
    NetworkSelectorDatasetInfo:mobilenet:se_ratio_7 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 6
    NetworkSelectorDatasetInfo:mobilenet:skip_con_1 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:skip_con_2 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:skip_con_3 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:skip_con_4 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 3
    NetworkSelectorDatasetInfo:mobilenet:skip_con_5 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 4
    NetworkSelectorDatasetInfo:mobilenet:skip_con_6 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 5
    NetworkSelectorDatasetInfo:mobilenet:skip_con_7 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 6
    NetworkSelectorDatasetInfo:mobilenet:stride_1 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:stride_2 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:stride_3 | NetworkSelectorDatasetInfo:network == 'mobilenet'
    NetworkSelectorDatasetInfo:mobilenet:stride_4 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 3
    NetworkSelectorDatasetInfo:mobilenet:stride_5 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 4
    NetworkSelectorDatasetInfo:mobilenet:stride_6 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 5
    NetworkSelectorDatasetInfo:mobilenet:stride_7 | NetworkSelectorDatasetInfo:mobilenet:nr_main_blocks > 6
    NetworkSelectorDatasetInfo:resnet:death_rate | NetworkSelectorDatasetInfo:network == 'resnet'
    NetworkSelectorDatasetInfo:resnet:initial_filters | NetworkSelectorDatasetInfo:network == 'resnet'
    NetworkSelectorDatasetInfo:resnet:nr_main_blocks | NetworkSelectorDatasetInfo:network == 'resnet'
    NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_1 | NetworkSelectorDatasetInfo:network == 'resnet'
    NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_2 | NetworkSelectorDatasetInfo:network == 'resnet'
    NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_3 | NetworkSelectorDatasetInfo:resnet:nr_main_blocks > 2
    NetworkSelectorDatasetInfo:resnet:nr_residual_blocks_4 | NetworkSelectorDatasetInfo:resnet:nr_main_blocks > 3
    NetworkSelectorDatasetInfo:resnet:res_branches_1 | NetworkSelectorDatasetInfo:network == 'resnet'
    NetworkSelectorDatasetInfo:resnet:res_branches_2 | NetworkSelectorDatasetInfo:network == 'resnet'
    NetworkSelectorDatasetInfo:resnet:res_branches_3 | NetworkSelectorDatasetInfo:resnet:nr_main_blocks > 2
    NetworkSelectorDatasetInfo:resnet:res_branches_4 | NetworkSelectorDatasetInfo:resnet:nr_main_blocks > 3
    NetworkSelectorDatasetInfo:resnet:widen_factor_1 | NetworkSelectorDatasetInfo:network == 'resnet'
    NetworkSelectorDatasetInfo:resnet:widen_factor_2 | NetworkSelectorDatasetInfo:network == 'resnet'
    NetworkSelectorDatasetInfo:resnet:widen_factor_3 | NetworkSelectorDatasetInfo:resnet:nr_main_blocks > 2
    NetworkSelectorDatasetInfo:resnet:widen_factor_4 | NetworkSelectorDatasetInfo:resnet:nr_main_blocks > 3
    OptimizerSelector:adamw:learning_rate | OptimizerSelector:optimizer == 'adamw'
    OptimizerSelector:adamw:weight_decay | OptimizerSelector:optimizer == 'adamw'
    OptimizerSelector:sgd:learning_rate | OptimizerSelector:optimizer == 'sgd'
    OptimizerSelector:sgd:momentum | OptimizerSelector:optimizer == 'sgd'
    OptimizerSelector:sgd:weight_decay | OptimizerSelector:optimizer == 'sgd'
    SimpleLearningrateSchedulerSelector:cosine_annealing:T_max | SimpleLearningrateSchedulerSelector:lr_scheduler == 'cosine_annealing'
    SimpleLearningrateSchedulerSelector:cosine_annealing:T_mult | SimpleLearningrateSchedulerSelector:lr_scheduler == 'cosine_annealing'
    SimpleLearningrateSchedulerSelector:step:gamma | SimpleLearningrateSchedulerSelector:lr_scheduler == 'step'
    SimpleLearningrateSchedulerSelector:step:step_size | SimpleLearningrateSchedulerSelector:lr_scheduler == 'step'
    SimpleTrainNode:mixup:alpha | SimpleTrainNode:batch_loss_computation_technique == 'mixup'
```

## Resulting Pytorch model
```Python
autoPyTorch.get_pytorch_model()
```
```
Sequential(
  (0): Sequential(
    (Conv_0): Conv2d(3, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (BN_0): BatchNorm2d(14, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Group_1): ResidualGroup(
      (group): Sequential(
        (Block_1): BasicBlock(
          (branches): ModuleList(
            (0): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_1:ReLU_1): ReLU()
                (Branch_1:Conv_1): Conv2d(14, 17, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_1:BN_1): BatchNorm2d(17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_1:ReLU_2): ReLU()
                (Branch_1:Conv_2): Conv2d(17, 17, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_1:BN_2): BatchNorm2d(17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (1): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_2:ReLU_1): ReLU()
                (Branch_2:Conv_1): Conv2d(14, 17, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_2:BN_1): BatchNorm2d(17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_2:ReLU_2): ReLU()
                (Branch_2:Conv_2): Conv2d(17, 17, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_2:BN_2): BatchNorm2d(17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (skip): Sequential(
            (Skip_connection): SkipConnection(
              (s1): Sequential(
                (Skip_1_AvgPool): AvgPool2d(kernel_size=1, stride=1, padding=0)
                (Skip_1_Conv): Conv2d(14, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
              )
              (s2): Sequential(
                (Skip_2_AvgPool): AvgPool2d(kernel_size=1, stride=1, padding=0)
                (Skip_2_Conv): Conv2d(14, 9, kernel_size=(1, 1), stride=(1, 1), bias=False)
              )
              (batch_norm): BatchNorm2d(17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (Block_2): BasicBlock(
          (branches): ModuleList(
            (0): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_1:ReLU_1): ReLU()
                (Branch_1:Conv_1): Conv2d(17, 17, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_1:BN_1): BatchNorm2d(17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_1:ReLU_2): ReLU()
                (Branch_1:Conv_2): Conv2d(17, 17, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_1:BN_2): BatchNorm2d(17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (1): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_2:ReLU_1): ReLU()
                (Branch_2:Conv_1): Conv2d(17, 17, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_2:BN_1): BatchNorm2d(17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_2:ReLU_2): ReLU()
                (Branch_2:Conv_2): Conv2d(17, 17, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_2:BN_2): BatchNorm2d(17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (skip): Sequential()
        )
      )
    )
    (Group_2): ResidualGroup(
      (group): Sequential(
        (Block_1): BasicBlock(
          (branches): ModuleList(
            (0): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_1:ReLU_1): ReLU()
                (Branch_1:Conv_1): Conv2d(17, 21, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (Branch_1:BN_1): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_1:ReLU_2): ReLU()
                (Branch_1:Conv_2): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_1:BN_2): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (1): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_2:ReLU_1): ReLU()
                (Branch_2:Conv_1): Conv2d(17, 21, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (Branch_2:BN_1): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_2:ReLU_2): ReLU()
                (Branch_2:Conv_2): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_2:BN_2): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (2): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_3:ReLU_1): ReLU()
                (Branch_3:Conv_1): Conv2d(17, 21, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (Branch_3:BN_1): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_3:ReLU_2): ReLU()
                (Branch_3:Conv_2): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_3:BN_2): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (skip): Sequential(
            (Skip_connection): SkipConnection(
              (s1): Sequential(
                (Skip_1_AvgPool): AvgPool2d(kernel_size=1, stride=2, padding=0)
                (Skip_1_Conv): Conv2d(17, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
              )
              (s2): Sequential(
                (Skip_2_AvgPool): AvgPool2d(kernel_size=1, stride=2, padding=0)
                (Skip_2_Conv): Conv2d(17, 11, kernel_size=(1, 1), stride=(1, 1), bias=False)
              )
              (batch_norm): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (Block_2): BasicBlock(
          (branches): ModuleList(
            (0): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_1:ReLU_1): ReLU()
                (Branch_1:Conv_1): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_1:BN_1): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_1:ReLU_2): ReLU()
                (Branch_1:Conv_2): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_1:BN_2): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (1): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_2:ReLU_1): ReLU()
                (Branch_2:Conv_1): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_2:BN_1): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_2:ReLU_2): ReLU()
                (Branch_2:Conv_2): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_2:BN_2): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (2): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_3:ReLU_1): ReLU()
                (Branch_3:Conv_1): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_3:BN_1): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_3:ReLU_2): ReLU()
                (Branch_3:Conv_2): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_3:BN_2): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (skip): Sequential()
        )
        (Block_3): BasicBlock(
          (branches): ModuleList(
            (0): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_1:ReLU_1): ReLU()
                (Branch_1:Conv_1): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_1:BN_1): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_1:ReLU_2): ReLU()
                (Branch_1:Conv_2): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_1:BN_2): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (1): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_2:ReLU_1): ReLU()
                (Branch_2:Conv_1): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_2:BN_1): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_2:ReLU_2): ReLU()
                (Branch_2:Conv_2): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_2:BN_2): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (2): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_3:ReLU_1): ReLU()
                (Branch_3:Conv_1): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_3:BN_1): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_3:ReLU_2): ReLU()
                (Branch_3:Conv_2): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_3:BN_2): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (skip): Sequential()
        )
        (Block_4): BasicBlock(
          (branches): ModuleList(
            (0): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_1:ReLU_1): ReLU()
                (Branch_1:Conv_1): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_1:BN_1): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_1:ReLU_2): ReLU()
                (Branch_1:Conv_2): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_1:BN_2): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (1): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_2:ReLU_1): ReLU()
                (Branch_2:Conv_1): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_2:BN_1): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_2:ReLU_2): ReLU()
                (Branch_2:Conv_2): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_2:BN_2): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (2): ResidualBranch(
              (residual_branch): Sequential(
                (Branch_3:ReLU_1): ReLU()
                (Branch_3:Conv_1): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_3:BN_1): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (Branch_3:ReLU_2): ReLU()
                (Branch_3:Conv_2): Conv2d(21, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (Branch_3:BN_2): BatchNorm2d(21, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
          )
          (skip): Sequential()
        )
      )
    )
    (ReLU_0): ReLU(inplace=True)
    (AveragePool): AdaptiveAvgPool2d(output_size=1)
  )
)
```