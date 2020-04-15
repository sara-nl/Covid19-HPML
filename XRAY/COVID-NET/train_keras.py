import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import multi_gpu_model
import tensorflow.keras
from model import build_COVIDNet
import pdb
import numpy as np
import os, pathlib, argparse
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from data import DataGenerator, BalanceDataGenerator, Metrics
from pprint import pprint
from sklearn import preprocessing
import glob
from shutil import copyfile
import collections

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # which gpu(s) to train on

# TO-DO: add argparse when converting to script
parser = argparse.ArgumentParser(description='COVID-Net Training')
parser.add_argument('--trainfile', default='train_COVIDx.txt', type=str, help='Name of train file')
parser.add_argument('--testfile', default='test_COVIDx.txt', type=str, help='Name of test file')
parser.add_argument('--data_path', default='data', type=str, help='Path to data folder')
parser.add_argument('--lr', default=0.00002, type=float, help='Learning rate')
parser.add_argument('--img_size', type=int, default=512, help='Image size to use')
parser.add_argument('--bs', default=8, type=int, help='Batch size')
parser.add_argument('--regularizer', action='store_true')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('--name', default='COVIDNet', type=str, help='Name of training folder')
parser.add_argument('--checkpoint', default='', type=str, help='Start training from existing weights')
parser.add_argument('--model', default='resnet50v2', type=str, help='Start training with model specification')
parser.add_argument('--datapipeline', default='covidx', type=str, help='Which data to use covidx | chexpert')
parser.add_argument('--val_split', default=0.1, type=float, help='What validation split to use')
 
args = parser.parse_args()
pprint(vars(args))

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
if args.datapipeline =='chexpert':
    class_weight = {0: 1.0,
                    1: 1.0,
                    2: 1.0,
                    3: 1.0,
                    4: 1.0,
                    5: 1.0,
                    6: 1.0,
                    7: 1.0,
                    8: 1.0,
                    9: 1.0,
                    10: 1.0,
                    11: 1.0,
                    12: 1.0,
                    13: 1.0,
                    14: 1.0}
elif args.datapipeline =='covidx':
    class_weight = {0: 1., 1: 1., 2: 25.}
num_classes = 3
batch_size = args.bs
epochs = args.epochs
lr = args.lr
outputPath = './output/'
runID = args.name + 'lr' + str(lr)
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

# load data
file = open(args.trainfile, 'r')
trainfiles = file.readlines()
file = open(args.testfile, 'r')
testfiles = file.readlines()




def get_data_references(images = [] , labels = [] , filter = "_positive.txt", copy_to = "/tmp/covid_dataset", balance = 180): #TODO: add balanced/imbalanced and number of examples limit 
    '''
    If you don't  want to copy set copy_to = None
    balance = 180 means each class gets 180 examples before splits
    '''
    for name in glob.glob("../data/*{}".format(filter)):
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
            dest = os.path.join(copy_to, f"{le.classes_[labels[i]].split('/')[-1]}-{str(i)}.{im.split('/')[-1].split('.')[-1]}")
            if not os.path.exists(dest):
                copyfile(im, dest)
            images[i] = dest
        print("Copy {} files".format(i+1))
    assert len(images) == len(labels)
    return images, labels, le

if args.datapipeline == 'chexpert':
    images, labels, le  = get_data_references()
    images, labels      = shuffle(images,random_state=0), shuffle(labels,random_state=0)
    
    train_images = images[:int(len(images)*(1-args.val_split))]
    train_labels = labels[:int(len(labels)*(1-args.val_split))]

    valid_images = images[int(len(images)*(1-args.val_split)):]
    valid_labels = labels[int(len(labels)*(1-args.val_split)):]
    
    tcounter=collections.Counter([x.split('-')[-2] for x in train_images])
    vcounter=collections.Counter([x.split('-')[-2] for x in valid_images])
    print("Train Images:\n"), pprint(tcounter), print("Valid Images:\n"), pprint(vcounter)

    
    train_generator = BalanceDataGenerator(trainfiles,
                                           images = train_images,
                                           labels = train_labels,
                                           le = le,
                                           input_shape=(args.img_size,args.img_size),
                                           datadir=args.data_path,
                                           is_training=True,
                                           args=args)
    test_generator = DataGenerator(testfiles,
                                   images = valid_images,
                                   labels = valid_labels,
                                   le = le,
                                   input_shape=(args.img_size,args.img_size),
                                   datadir=args.data_path,
                                   is_training=False,
                                   args=args)


elif args.datapipeline == 'covidx':
    train_generator = BalanceDataGenerator(trainfiles,
                                           input_shape=(args.img_size,args.img_size),
                                           datadir=args.data_path,
                                           is_training=True,
                                           args=args)
    test_generator = DataGenerator(testfiles,
                                   input_shape=(args.img_size,args.img_size),
                                   datadir=args.data_path,
                                   is_training=False,
                                   args=args)


def get_callbacks(runPath):
    callbacks = []
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=0.000001, min_delta=1e-2)
    callbacks.append(lr_schedule) # reduce learning rate when stuck

    checkpoint_path = runPath + '/cp-{epoch:02d}-{val_loss:.2f}.hdf5'
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
        verbose=1, save_best_only=False, save_weights_only=True, mode='min', period=1))

    class SaveAsCKPT(tf.keras.callbacks.Callback):
        def __init__(self):
            self.saver = tf.train.Saver()
            self.sess = tf.keras.backend.get_session()

        def on_epoch_end(self, epoch, logs=None):
            checkpoint_path = runPath + '/cp-{:02d}.ckpt'.format(epoch)
            save_path = self.saver.save(self.sess, checkpoint_path)
    callbacks.append(SaveAsCKPT())
    
    metrics = Metrics(validation_generator=test_generator)
    callbacks.append(metrics)
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=checkpoint_path))

    return callbacks


model = build_COVIDNet(checkpoint=args.checkpoint,args=args,num_classes=train_generator.n_classes)

if args.regularizer:
    # Setting L2 regularization
    for layer in model.layers:
        if hasattr(layer,'kernel_regularizer'):
            layer.kernel_regularizer = tf.keras.regularizers.l2(l=1e-3)
            print("      Reg: ",layer.kernel_regularizer )
        
opt = Adam(learning_rate=lr, amsgrad=True)
callbacks = get_callbacks(runPath)

paramodel= multi_gpu_model(model, gpus=4)
paramodel.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy']) # TO-DO: add additional metrics for COVID-19

paramodel.summary()
print('Ready for training!')

paramodel.fit_generator(train_generator, 
                    callbacks=callbacks, 
                    validation_data=test_generator, 
                    epochs=epochs, 
                    shuffle=False, 
                    class_weight=class_weight, 
                    use_multiprocessing=False)

# model.fit(train_generator, 
#                     callbacks=callbacks, 
#                     validation_data=test_generator, 
#                     epochs=epochs, 
#                     shuffle=True, 
#                     class_weight=class_weight, 
#                     use_multiprocessing=False)

y_test = []
pred = []
for i in range(len(testfiles)):
    line = testfiles[i].split()
    x = cv2.imread(os.path.join(args.data_path, 'test', line[1]))
    x = cv2.resize(x, (args.img_size, args.img_size))
    x = x.astype('float32') / 255.0
    y_test.append(mapping[line[2]])
    pred.append(np.array(model.predict(np.expand_dims(x, axis=0))).argmax(axis=1))
y_test = np.array(y_test)
pred = np.array(pred)

matrix = confusion_matrix(y_test, pred)
matrix = matrix.astype('float')
print(matrix)
class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                           class_acc[1],
                                                                           class_acc[2]))
ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                         ppvs[1],
                                                                         ppvs[2]))
