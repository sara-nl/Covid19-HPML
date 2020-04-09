import glob
import os
from shutil import copyfile
from pprint import pprint
import pickle
from PIL import Image
import numpy as np
from autoPyTorch import AutoNetClassification, HyperparameterSearchSpaceUpdates
import sklearn.model_selection
import sklearn.metrics
from sklearn import preprocessing

input_size = (256,256)

def get_data_references(images = [] , labels = [] , filter = "_positive.txt", copy_to = "/tmp/covid_dataset", balance = 180): #TODO: add balanced/imbalanced and number of examples limit 
    '''
    If you don't  want to copy set copy_to = None
    balance = 180 means each class gets 180 examples before splits
    '''
    for name in glob.glob("../XRAY/*{}".format(filter)):
        for i, imagepaths in enumerate(open(name).readlines()):
            if i==balance:
                break
            images.append(imagepaths)
            labels.append(name.split(filter)[0])
    le = preprocessing.LabelEncoder()
    le.fit(list(set(labels)))
    #Create sklearn type labels
    labels = le.transform(labels)
    print("Found {} examples with labels {}".format(len(labels), le.classes_))
    #Copy data locally if needed
    if copy_to:
        os.makedirs(copy_to, exist_ok=True)
        for i, im in enumerate(images):
            if "\n" in im:
                im = im.strip()
            dest = os.path.join(copy_to, "{0}.{1}".format(str(i), im.split("/")[-1].split(".")[-1]))
            copyfile(im, dest)
            images[i] = dest
        print("Copy {} files".format(i+1))
    assert len(images) == len(labels)
    return images, labels, le

def jpg_image_to_array(image_path):
    '''
    Loads JPEG imag
    '''
    with Image.open(image_path).convert("RGB") as image:
        image = image.resize(input_size)
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8) / 255.0
        try:
            im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
        except ValueError as e:
            im_arr = im_arr.reshape((image.size[1], image.size[0]))
            im_arr = np.stack((im_arr,)*3, axis=-1)
    return im_arr

if __name__ == "__main__":
    #Get the dataset (this could be yielded/batched)
    images, labels, le  = get_data_references()
    #Read the images
    X = []
    for i, im in enumerate(images):
        image = jpg_image_to_array(im)
        X.append(image)
    #autopytorch format
    X = [np.asarray(x).flatten() for x in X]
    #Split the dataset
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(np.asarray(X, dtype=np.float16), np.asarray(labels, dtype=np.uint8), random_state=1, test_size=0.1, shuffle=True)
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #Init custom search space
    # search_space_updates = HyperparameterSearchSpaceUpdates()
    # search_space_updates.append(node_name="CreateDataLoader",
    #                             hyperparameter="batch_size",
    #                             value_range=[32],
    #                             log=False)
    #Init autonet
    # autoPyTorch = AutoNetClassification(hyperparameter_search_space_updates=search_space_updates,  # config 
    autoPyTorch = AutoNetClassification(
                                        # "full_cs",  # config 
                                        networks=["resnet"],
                                        # torch_num_threads=2, 
                                        log_level='info',
                                        budget_type='epochs',
                                        min_budget=5,
                                        max_budget=20,
                                        num_iterations=100,
                                        cuda=True, use_pynisher=False)
    #fit
    autoPyTorch.fit(X_train=X_train, Y_train=y_train, X_valid=X_test, Y_valid=y_test, optimize_metric="auc_metric", loss_modules=["cross_entropy", "cross_entropy_weighted"])
    #predict
    y_pred = autoPyTorch.predict(X_test)
    #check
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))
    print("Confusion matrix", sklearn.metrics.confusion_matrix(y_test, y_pred, labels=le.classes_))
    print(autoPyTorch)
    pytorch_model = autoPyTorch.get_pytorch_model()
    print(pytorch_model)