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

input_size = (512,512)
classes = ["covid", "Pneumonia", "Cardiomegaly", "Consolidation", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pneumothorax"]

def get_data_references(images = [] , labels = [] , filter = "_positive.txt", copy_to = "/tmp/covid_dataset", balance = 280, classes=classes): #TODO: add balanced/imbalanced and number of examples limit 
    '''
    If you don't  want to copy set copy_to = None
    balance = 180 means each class gets 180 examples before splits
    '''
    for name in glob.glob("../data/*{}".format(filter)):
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
            with Image.open(im).convert("RGB") as image:
                image = image.resize(input_size)
                im_arr = np.fromstring(image.tobytes(), dtype=np.uint8) / 255.0
            try:
                im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
            except ValueError as e:
                im_arr = im_arr.reshape((image.size[1], image.size[0]))
                im_arr = np.stack((im_arr,)*3, axis=-1)
            finally:
                dest = os.path.join(copy_to, "{0}.{1}".format(str(i), im.split("/")[-1].split(".")[-1]))
                # copyfile(im, dest)
                image.save(dest) 
            images[i] = dest
        print("Copy {} files".format(i+1))
    assert len(images) == len(labels)
    return images, labels, le


if __name__ == "__main__":
    #Get the dataset (this could be yielded/batched)
    images, labels, le  = get_data_references()
    autoPyTorch = AutoNetImageClassification(
                                        # "tiny_cs",  # config 
                                        # networks=["resnet"],
                                        # torch_+num_threads=40,
                                        min_workers=4, 
                                        log_level='info',
                                        budget_type='epochs',
                                        min_budget=1,
                                        max_budget=60,
                                        num_iterations=100, #magical lipschitz value = 20 updates / stochastic process
                                        cuda=True)
    # search_space_updates = HyperparameterSearchSpaceUpdates()
    #fit
    autoPyTorch.fit(images, labels, optimize_metric="balanced_accuracy", use_tensorboard_logger=True, loss_modules=['cross_entropy'], validation_split=0.1)
    # autoPyTorch.fit(images, labels, optimize_metric="accuracy", use_tensorboard_logger=True, networks=['resnet'], lr_scheduler=['cosine_annealing'], batch_loss_computation_techniques=['mixup'], loss_modules=['cross_entropy'], optimizer=['adamw'], validation_split=0.1)
    import ipdb; ipdb.set_trace()
    print(autoPyTorch)
    print(autoPyTorch.get_pytorch_model())