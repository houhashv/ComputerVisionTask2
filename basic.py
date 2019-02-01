from keras.applications import VGG16
from keras import models
from keras import layers, Input
from keras import optimizers
import scipy.io
import numpy as np
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten
from keras.utils import np_utils
import time

from keras.models import Sequential
from keras.layers import Dense

from hyperas import optim
from hyperas.distributions import choice, uniform
import multiprocessing
import tensorflow as tf
from keras import backend as backend


def configure(num_cores=None,gpu=False):

    if ~num_cores:
        num_cores = multiprocessing.cpu_count()

    if gpu:
        num_gpu= 1
        num_cpu = 1
    else:
        num_cpu = 1
        num_gpu = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True, \
                            device_count = {'CPU' : num_cpu, 'GPU' : num_gpu})
    session = tf.Session(config=config)
    backend.set_session(session)

def resizeImages(images, resize_factor):

    images_resize_array = []
    for img in images[0, 0:]:
        resize_img = cv2.resize(img, (resize_factor, resize_factor))
        resize_img = np.expand_dims(resize_img, axis=0)
        resize_img = preprocess_input(resize_img)
        images_resize_array.append(resize_img)
    images_resize_array=np.array(images_resize_array)
    images_resize_array=np.rollaxis(images_resize_array,1,0)
    images_resize_array=images_resize_array[0]
    return images_resize_array


def divideToTrainTest(resize_images,Y,k):

    train_set=[]
    test_set=[]
    y_train=[]
    y_test=[]
    for i in range(0,len(resize_images)):
        if(i<k):
            train_set.append(resize_images[i])
            y_train.append(Y[i])
        else:
            test_set.append(resize_images[i])
            y_test.append(Y[i])
    return np.array(train_set), np.array(test_set), np.array(y_train), np.array(y_test)


if __name__ == '__main__':

    data_path = "C:/Users/yossi/OneDrive/Desktop/FlowerData/FlowerDataLabels.mat"
    # reading flower images from file
    data = scipy.io.loadmat(data_path)
    images = data['Data']
    labels = data['Labels']
    num_classes = 2
    Y = np_utils.to_categorical(labels[0], num_classes)
    resized_imgs = resizeImages(images, 224)
    train_set, test_set,y_train,y_test= divideToTrainTest(resized_imgs,Y,300)
    #Load the VGG model
    image_input = Input(shape=(224, 224, 3))
    """
    model_vgg = VGG16(weights='imagenet', include_top=True, input_tensor=image_input)
    # model.load_weights(‘my_weights’)
    last_layer = model_vgg.get_layer('block5_pool').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    out = Dense(num_classes, activation='softmax', name='output')(x)
    model_vgg_new = Model(image_input, out)
    model_vgg_new.summary()
    # freeze all the layers except the dense layers
    for layer in model_vgg_new.layers[:-3]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in model_vgg_new.layers:
        print(layer, layer.trainable)
    """

    # create model
    import requests
    import pandas as pd
    import io
    import numpy
    from keras.utils import to_categorical
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    s = requests.get(url).content
    dataset = pd.read_csv(io.StringIO(s.decode('utf-8'))).values
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    model = Sequential()
    model.add(Dense(1024, input_dim=8, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(214, activation='relu'))
    model.add(Dense(124, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    adam = optimizers.Adam(lr=0.00001)
    SGD = optimizers.SGD(lr=0.00001)
    model_vgg_new = model
    model_vgg_new.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    t = time.time()
    import pickle
    pickle.dump(model_vgg_new, open("save.p", "wb"))
    model_vgg_new = pickle.load(open("save.p", "rb"))
    # h = model_vgg_new.fit(train_set, y_train, batch_size=2, epochs=10, verbose=1, validation_split=0.2)
    h = model_vgg_new.fit(X, to_categorical(Y), batch_size=50, epochs=100, verbose=1, validation_split=0.2)
    pickle.dump(h, open("save.p", "wb"))
    h = pickle.load(open("save.p", "rb"))
    h = model_vgg_new.fit(X, to_categorical(Y), batch_size=50, epochs=100, verbose=1, validation_split=0.2)
    import pickle


    print('Training time: %s' % (time.time() - t))

    # (loss, accuracy) = model_vgg_new.evaluate(test_set, y_test, batch_size=2, verbose=1)

    # print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
