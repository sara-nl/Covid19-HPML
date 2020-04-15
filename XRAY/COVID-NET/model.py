from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os 
import pdb
import sys
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, Activation, Dropout,GlobalMaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D


def pep_x_module(inputs, block_id, filter_size=[64,64,128]):

    x = Conv2D(filter_size[0], kernel_size=(1,1), padding='valid')(inputs)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    
    x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    
    x = Conv2D(filter_size[1], kernel_size=(1,1), padding='valid')(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filter_size[2], kernel_size=(1,1), padding='valid')(x)

    return x


def covidnet(input_tensor=None, input_shape=(224, 224, 3), classes=3):
    """ Instantiates the covidnet architecture
    https://github.com/lindawangg/COVID-Net/blob/master/assets/COVID_Netv2.pdf
    
    """


    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor
    
    

    x = Conv2D(64, (7, 7), strides=(2, 2),name='conv1_1', padding='valid')(img_input)

    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='valid')(x)
    
    x = BatchNormalization(epsilon=1e-5,name='entry_flow_conv1_2_BN')(x)
    x = Activation('relu')(x)
    
    add_1 = Conv2D(256, kernel_size=(1,1), padding='valid', name='conv3')(x)   
    add_2 = pep_x_module(x, 'PEP1.1/',filter_size=[64,64,256])
    x1 = tf.keras.layers.Add()([add_1, add_2])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP1.2/',filter_size=[64,64,256])
    x1 = tf.keras.layers.Add()([x1, add_1])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP1.3/',filter_size=[64,64,256])
    add_2 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1,1), padding='valid')(x1)
    x = tf.keras.layers.Add()([add_1, add_2])
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP2.1/',filter_size=[128,128,512])
    add_2 = Conv2D(512, kernel_size=(1,1), padding='valid')(x)   
    x1 = tf.keras.layers.Add()([add_1, add_2])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP2.2/',filter_size=[128,128,512])
    x1 = tf.keras.layers.Add()([add_1, x1])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP2.3/',filter_size=[128,128,512])
    x1 = tf.keras.layers.Add()([add_1, x1])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP2.4/',filter_size=[128,128,512])
    add_2 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1,1), padding='valid')(x1)
    x = tf.keras.layers.Add()([add_1, add_2])
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP3.1/',filter_size=[256,256,1024])
    add_2 = Conv2D(1024, kernel_size=(1,1), padding='valid')(x)   
    x1 = tf.keras.layers.Add()([add_1, add_2])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP3.2/',filter_size=[256,256,1024])
    x1 = tf.keras.layers.Add()([add_1, x1])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP3.3/',filter_size=[256,256,1024])
    x1 = tf.keras.layers.Add()([add_1, x1])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP3.4/',filter_size=[256,256,1024])
    x1 = tf.keras.layers.Add()([add_1, x1])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP3.5/',filter_size=[256,256,1024])
    x1 = tf.keras.layers.Add()([add_1, x1])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP3.6/',filter_size=[256,256,1024])
    add_2 = tf.keras.layers.MaxPool2D(pool_size=(1,1), strides=(1,1), padding='valid')(x1)
    x = tf.keras.layers.Add()([add_1, add_2])
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP4.1/',filter_size=[512,512,2048])
    add_2 = Conv2D(2048, kernel_size=(1,1), padding='valid')(x)   
    x1 = tf.keras.layers.Add()([add_1, add_2])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP4.2/',filter_size=[512,512,2048])
    x1 = tf.keras.layers.Add()([add_1, x1])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    
    add_1 = pep_x_module(x, 'PEP4.3/',filter_size=[512,512,2048])
    x1 = tf.keras.layers.Add()([add_1, x1])
    x = BatchNormalization(epsilon=1e-5)(x1)
    x = Activation('relu')(x)
    

    model = Model(img_input, x, name='covidnetpaper')
    
    return model

def build_COVIDNet(num_classes=3, flatten=True, checkpoint='',args=None):
    
    if args.model == 'resnet50v2':
        base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(args.img_size, args.img_size, 3))
        x = base_model.output
    
    if args.model =='mobilenetv2':
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(args.img_size, args.img_size, 3))
        x = base_model.output
    
    if args.model == 'custom':
        base_model = covidnet(input_tensor=None, input_shape=(args.img_size, args.img_size, 3), classes=3)
        x = base_model.output
        
    if args.model == 'EfficientNet':
        import efficientnet.tfkeras as efn
        base_model = efn.EfficientNetB4(weights=None, include_top=True, input_shape=(args.img_size, args.img_size, 3), classes=3)
        x = base_model.output
    
    
    if flatten:
        x = Flatten()(x)
    else:
        # x = GlobalAveragePooling2D()(x)
        x = GlobalMaxPool2D()(x)
    
    if args.datapipeline == 'covidx':
        x = Dense(1024, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    # x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax',name=f'FC_{num_classes}')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    if len(checkpoint):
        model.load_weights(checkpoint)
    return model