import glob
import os
# from shutil import copyfile
from pprint import pprint
import pickle
from PIL import Image
import numpy as np
from autoPyTorch import AutoNetImageClassification, HyperparameterSearchSpaceUpdates
import sklearn.model_selection
import sklearn.metrics
from sklearn import preprocessing
import torch
from torch.autograd import Variable
from torchvision import transforms
import json

input_size = (8, 8)
channels = 3
balance = 280 #Examples / class, set None if you want to load all examples
final_dataset_location = "/tmp/covid_dataset"
initial_dataset_location = "../data"
classes = ["covid", "Pneumonia", "No Finding"]
preset = "full_cs"
save_output_to = "/tmp/{2}class_{1}balanced_ba_ce_{0}_{3}".format(input_size, balance, len(classes), preset)


def get_data_references(images = [] , labels = [] , filter = "_positive.txt", copy_to = final_dataset_location, copy_from = initial_dataset_location, balance = balance, classes = classes, img_channels = channels):
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
    le = preprocessing.LabelBinarizer()
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


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU


def jpg_image_to_array(image_path):
    """
    Loads JPEG imag
    """
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
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(images, labels, test_size=0.1, random_state=9, shuffle=True)
    search_space_updates = HyperparameterSearchSpaceUpdates()
    #TODO: this still runs out of memory and wastes resources
    search_space_updates.append(node_name="CreateImageDataLoader", hyperparameter="batch_size", log=False, \
                                value_range=[2, int(len(labels)/20)]) #Lipschitz magical number
    autonet = AutoNetImageClassification(
                                        preset, \
                                        hyperparameter_search_space_updates=search_space_updates, \
                                        min_workers=2, \
                                        dataloader_worker=4, \
                                        global_results_dir="results", \
                                        keep_only_incumbent_checkpoints=False, \
                                        log_level="info", \
                                        budget_type="epochs", \
                                        save_checkpoints=True, \
                                        result_logger_dir=save_output_to, \
                                        min_budget=1, \
                                        max_budget=10, \
                                        num_iterations=1, \
                                        images_shape=[channels, input_size[0], input_size[1]], \
                                        cuda=True \
                                        )

    #fit
    
    # autoPyTorch.fit(images, labels, optimize_metric="balanced_accuracy", use_tensorboard_logger=True, loss_modules=['cross_entropy'], validation_split=0.1)
    # autoPyTorch.fit(images, labels, optimize_metric="accuracy", use_tensorboard_logger=True, networks=['resnet'], lr_scheduler=['cosine_annealing'], batch_loss_computation_techniques=['mixup'], loss_modules=['cross_entropy'], validation_split=0.1)
    # autoPyTorch.fit(images, labels, optimize_metric="balanced_accuracy",  networks=['densenet', 'densenet_flexible', 'resnet', 'resnet152', 'darts'], use_tensorboard_logger=True, loss_modules=["cross_entropy"], validation_split=0.1)

    results_fit = autonet.fit(\
                            X_train=X_train, Y_train=y_train, \
                            X_valid=X_test, Y_valid=y_test, \
                            optimizer = ["adam", "adamw", "sgd", "rmsprop"], \
                            algorithm="hyperband", \
                            optimize_metric="balanced_accuracy", \
                            additional_metrics=["pac_metric"], \
                            lr_scheduler=["cosine_annealing", "cyclic", "step", "adapt", "plateau", "alternating_cosine", "exponential"], \
                            networks=['resnet','densenet'], #, 'densenet_flexible', 'resnet', 'resnet152', 'darts'], \
                            refit=True, \
                            use_tensorboard_logger=True \
                            )

    with open("{}/results_fit.json".format(save_output_to), "w") as file:
        json.dump(results_fit, file)


    import ipdb; ipdb.set_trace()
    autonet_config=autonet.get_current_autonet_config()
    with open("{}/autonet_config.json".format(save_output_to), "w") as file:
        json.dump(autonet_config, file)


    # X_test = [jpg_image_to_array(im) for im in X_test]
    print(autonet.score(X_test, y_test[:len(X_test)-3]))



    import ipdb; ipdb.set_trace()

    print("autoPyTorch.autonet_config")
    print(autoPyTorch.autonet_config)
    print("autoPyTorch.get_hyperparameter_search_space()")
    print(autoPyTorch.get_hyperparameter_search_space())
    print("autoPyTorch.get_pytorch_model()")
    print(autoPyTorch.get_pytorch_model())


    import ipdb; ipdb.set_trace()

    # loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
    # model_state = torch.load("/tmp/3class_100balanced_ba_ce_(16, 16)/checkpoints/checkpoint_(0, 0, 0)_Budget_2.pt")
    # # import ipdb; ipdb.set_trace()
    # model = model['state']
    # model.to(torch.device('cuda'))
    # image = image_loader(images[0])
    # print(model(image))

    model = autoPyTorch.get_pytorch_model()
    model_state = torch.load("{}/checkpoints/checkpoint_(6, 0, 0)_Budget_2.pt".format(save_output_to))['state'] #How do you know what checkpoint to load?
    model.load_state_dict(model_state)
    torch.save(model, "{}/torch_checkpoint".format(save_output_to))
    testmodel = torch.load("{}/torch_checkpoint".format(save_output_to))