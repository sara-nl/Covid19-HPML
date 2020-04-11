import glob
import os
from shutil import copyfile
from pprint import pprint
import pickle
from PIL import Image
import numpy as np
from autoPyTorch import AutoNetImageClassification, HyperparameterSearchSpaceUpdates
import sklearn.model_selection
import sklearn.metrics
from sklearn import preprocessing

input_size = (256, 256)
channels = 3
final_dataset_location = "/tmp/covid_dataset"
initial_dataset_location = "../data"
save_output_to = "3class_280balanced_ba_ce"
classes = ["covid", "Pneumonia", "No Finding"]


def get_data_references(images = [] , labels = [] , filter = "_positive.txt", copy_to = final_dataset_location, copy_from = initial_dataset_location, balance = 280, classes=classes, img_channels = channels):
    '''
    If you don't  want to copy set copy_to = None; only if this is set img_channels is taken into account 
    balance = 180 means each class gets 180 examples before splits
    '''
    for name in glob.glob(initial_dataset_location + os.path.sep + "*{}".format(filter)):
        class_name = name.split(filter)[0].split("/")[-1].lower()
        if any(class_name in s.lower() for s in classes):
            for i, imagepaths in enumerate(open(name).readlines()):
                if i==balance:
                    break
                images.append(imagepaths)
                labels.append(class_name)
    le = preprocessing.LabelEncoder()
    le.fit(list(set(labels)))
    #Create sklearn type labels
    labels = le.transform(labels)
    print("Found {} examples with labels {}".format(len(labels), le.classes_))
    assert len(labels) > 0, "No data found"
    #Copy data locally and preprocess
    if copy_to:
        os.makedirs(copy_to, exist_ok=True)
        for i, im in enumerate(images):
            if "\n" in im:
                im = im.strip()
            if img_channels == 3:
                mode = "RGB"
            elif img_channels == 1:
                mode = "L"
            with Image.open(im).convert(mode) as image:
                image = image.resize(input_size)
                im_arr = np.fromstring(image.tobytes(), dtype=np.uint8) / 255.0
            try:
                im_arr = im_arr.reshape((image.size[1], image.size[0], img_channels))
            except ValueError as e:
                im_arr = im_arr.reshape((image.size[1], image.size[0]))
                im_arr = np.stack((im_arr,) * img_channels, axis=-1)
            finally:
                im_arr = np.moveaxis(im_arr, -1, 0) #Pytorch is channelfirst
                dest = os.path.join(copy_to, "{0}.{1}".format(str(i), im.split("/")[-1].split(".")[-1]))
                # copyfile(im, dest)
                image.save(dest) 
                # print("Wrote image {} of shape {} with label {}({})".format(dest, im_arr.shape, labels[i], le.inverse_transform(labels)[i]))
            images[i] = dest
        print("Copy {} files".format(i+1))
    assert len(images) == len(labels)
    return images, labels, le


if __name__ == "__main__":
    #Get the dataset (this could be yielded/batched)
    images, labels, le  = get_data_references()
    search_space_updates = HyperparameterSearchSpaceUpdates()
    #TODO: this still runs out of memory and wastes resources
    search_space_updates.append(node_name="CreateImageDataLoader", hyperparameter="batch_size", log=False, \
                                value_range=[2, int(len(labels)/20)]) #Lipschitz magical number
    autoPyTorch = AutoNetImageClassification(
                                        "full_cs", \
                                        hyperparameter_search_space_updates=search_space_updates, \
                                        min_workers=2, \
                                        dataloader_worker=4, \
                                        global_results_dir="results", \
                                        keep_only_incumbent_checkpoints=True, \
                                        log_level="info", \
                                        budget_type="epochs", \
                                        save_checkpoints=True, \
                                        result_logger_dir=save_output_to, \
                                        min_budget=1, \
                                        max_budget=60, \
                                        num_iterations=5, \
                                        images_shape=[channels, input_size[0], input_size[1]], \
                                        cuda=True \
                                        )
    #fit
    # autoPyTorch.fit(images, labels, use_tensorboard_logger=True, validation_split=0.1)
    # autoPyTorch.fit(images, labels, optimize_metric="balanced_accuracy", use_tensorboard_logger=True, loss_modules=['cross_entropy'], validation_split=0.1)
    # autoPyTorch.fit(images, labels, optimize_metric="accuracy", use_tensorboard_logger=True, networks=['resnet'], lr_scheduler=['cosine_annealing'], batch_loss_computation_techniques=['mixup'], loss_modules=['cross_entropy'], validation_split=0.1)
    autoPyTorch.fit(images, labels, optimize_metric="balanced_accuracy",  networks=['densenet', 'densenet_flexible', 'resnet', 'resnet152', 'darts'], use_tensorboard_logger=True, loss_modules=["cross_entropy"], validation_split=0.1)
    print("autoPyTorch.get_autonet_config_file_parser().print_help()")
    print(autoPyTorch.get_autonet_config_file_parser().print_help())
    print("autoPyTorch.get_hyperparameter_search_space()")
    print(autoPyTorch.get_hyperparameter_search_space())
    print("autoPyTorch.get_pytorch_model()")
    print(autoPyTorch.get_pytorch_model())