from keras.applications import VGG16
from keras import Input
from keras.models import Model
from keras import optimizers
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Flatten
from keras.utils import np_utils
from keras import backend as backend
from keras.layers import Dense
import multiprocessing
import tensorflow as tf
import scipy.io
import numpy as np
import cv2
import time


def configure(num_cores=None, gpu_ind=False):

    if ~num_cores:
        num_cores = multiprocessing.cpu_count()

    if gpu_ind:
        num_gpu = 1
        num_cpu = 1
    else:
        num_cpu = 1
        num_gpu = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True, \
                            device_count = {'CPU' : num_cpu, 'GPU' : num_gpu})
    session = tf.Session(config=config)
    backend.set_session(session)


def resize_images(images, resize_factor):

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


def divide_to_train_test(resize_images, Y, k):

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

    gpu = False
    configure(gpu_ind=gpu)
    data_path = "C:/Users/yossi/OneDrive/Desktop/FlowerData/FlowerDataLabels.mat"
    # reading flower images from file
    data = scipy.io.loadmat(data_path)
    images = data['Data']
    labels = data['Labels']
    num_classes = 2
    Y = np_utils.to_categorical(labels[0], num_classes)
    resized_imgs = resize_images(images, 224)
    train_set, test_set,y_train,y_test= divide_to_train_test(resized_imgs, Y, 300)
    #Load the VGG model
    image_input = Input(shape=(224, 224, 3))
    model_vgg = VGG16(weights='imagenet', include_top=True, input_tensor=image_input)
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

    adam = optimizers.Adam(lr=0.00001)
    SGD = optimizers.SGD(lr=0.00001)
    model_vgg_new.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    h = model_vgg_new.fit(X, Y, batch_size=50, epochs=100, verbose=1, validation_split=0.2)


    print('Training time: %s' % (time.time() - t))

    # (loss, accuracy) = model_vgg_new.evaluate(test_set, y_test, batch_size=2, verbose=1)

    # print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
