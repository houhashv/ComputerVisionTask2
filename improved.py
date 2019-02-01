from keras.applications import VGG16
from keras import models
from keras import layers,Input
from keras.models import Model
from keras import optimizers
import scipy.io
import numpy as np
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten
from keras.utils import np_utils
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform
import tensorflow as tf
from keras import backend as K
import multiprocessing
import time


num_cores = multiprocessing.cpu_count()
GPU = True

if GPU:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


def resizeImages(images, resize_factor):

    images_resize_array = []
    for img in images[0, 0:]:
        resize_img=cv2.resize(img, (resize_factor, resize_factor))
        resize_img=np.expand_dims(resize_img, axis=0)
        resize_img=preprocess_input(resize_img)
        images_resize_array.append(resize_img)
    images_resize_array=np.array(images_resize_array)
    images_resize_array=np.rollaxis(images_resize_array,1,0)
    images_resize_array=images_resize_array[0]
    return images_resize_array

def data():

    data_path = 'C:/Users/yossi/OneDrive/Desktop/FlowerData/FlowerDataLabels.mat'
    # reading flower images from file
    data = scipy.io.loadmat(data_path)
    images = data['Data']
    labels = data['Labels']
    num_classes = 2
    Y = np_utils.to_categorical(labels[0], num_classes)
    images_resize_array = []
    for img in images[0, 0:]:
        resize_img = cv2.resize(img, (224, 224))
        resize_img = np.expand_dims(resize_img, axis=0)
        resize_img = preprocess_input(resize_img)
        images_resize_array.append(resize_img)
    images_resize_array = np.array(images_resize_array)
    images_resize_array = np.rollaxis(images_resize_array, 1, 0)
    resize_images = images_resize_array[0]

    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    k = 300
    for i in range(0,len(resize_images)):
        if(i<k):
            x_train.append(resize_images[i])
            y_train.append(Y[i])
        else:
            x_test.append(resize_images[i])
            y_test.append(Y[i])
    return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)


def create_model(x_train,y_train,x_test,y_test):

    image_input = Input(shape=(224, 224, 3))
    model_vgg = VGG16(weights='imagenet', include_top=True, input_tensor=image_input)
    # model_vgg = convnet('alexnet', weights_path="weights/alexnet_weights.h5", heatmap=False)
    last_layer = model_vgg.get_layer('block5_pool').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    out = Dense(2, activation='softmax', name='output')(x)
    model_vgg_new = Model(image_input, out)
    #model_vgg_new.summary()
    # freeze all the layers except the dense layers
    for layer in model_vgg_new.layers[:-3]:
        layer.trainable = False

    adam = optimizers.Adam(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    rmsprop = optimizers.RMSprop(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    sgd = optimizers.SGD(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'rmsprop':
        optim = rmsprop
    else:
        optim = sgd

    model_vgg_new.compile(loss='categorical_crossentropy', optimizer=optim,
                          metrics=['accuracy'])
    h = model_vgg_new.fit(np.array(x_train), np.array(y_train), batch_size=1, epochs=1, verbose=1, validation_split=0.2)
    score, acc = model_vgg_new.evaluate(np.array(x_test), np.array(y_test), verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model_vgg_new}


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = data()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

