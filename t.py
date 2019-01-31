from keras.applications import VGG16
from keras import models
from keras import layers, Input
from keras.models import Model
from keras import optimizers
import scipy.io
import numpy as np
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten
from keras.utils import np_utils
import time
from hyperas import optim
from hyperas.distributions import choice, uniform


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
    adam = optimizers.Adam(lr=0.00001)
    SGD = optimizers.SGD(lr=0.00001)
    model_vgg_new.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    t = time.time()
    h = model_vgg_new.fit(train_set, y_train, batch_size=32, epochs=1, verbose=1, validation_split=0.2)

    print('Training time: %s' % (t - time.time()))

    (loss, accuracy) = model_vgg_new.evaluate(test_set, y_test, batch_size=10, verbose=1)

    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
