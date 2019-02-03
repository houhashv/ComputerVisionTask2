from keras import Input
from keras.applications import VGG16
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Flatten, Dropout, Dense, TimeDistributed, MaxPooling2D
from keras import backend as backend
from keras import optimizers
from keras import regularizers
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import precision_recall_curve, accuracy_score, log_loss
from sklearn.utils.fixes import signature
import multiprocessing
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import pickle
from datetime import datetime
import os
import random
from numba import cuda
import json


"""
Task 2 - task transfer 
Idan Weiss and Yossi Hohashvili

if u want to see the results in tensorboard
go to the terminal and run: tensorboard --logdir=logs\
and them open the browser and type: localhost:6006
and you'll see it

"""

def set_params():
    """
    setting up all the configuration parameters and hyper parameters for the program
    :return: a dictionary of all the configurations
    """
    parameters = dict(hyper_parameters={})
    parameters["path"] = os.getcwd() + "/FlowerData/FlowerDataLabels.mat"
    parameters["optimizer"] = ["Adam"]
    parameters["num_classes"] = 1
    parameters["split_percent"] = 0.25
    parameters["first_n_split"] = 300
    parameters["seed"] = 0
    parameters["size"] = 224
    parameters["hyper_parameters"]["learning_rate"] = [10 ** -i for i in range(1, 4)]
    parameters["hyper_parameters"]["batch_size"] = [32]
    parameters["hyper_parameters"]["epochs"] = [10]
    parameters["hyper_parameters"]["level"] = {"basic": {}, "imprvoed": {}}
    parameters["hyper_parameters"]["basic"] = {"dropouts": [0], "l1": [0], "l2": [0]}
    parameters["hyper_parameters"]["improved"] = {"dropouts": [x / 10 for x in range(2, 8)],
                                                  "l1": [10 ** -i for i in range(0, 3)],
                                                  "l2": [10 ** -i for i in range(0, 3)]}
    return parameters


def configure(num_cores=None, gpu_ind=False):
    """
    configure the setting of the tensorflow to work with a GPU or CPU
    :param num_cores: the number of cores to use in parallel
    :param gpu_ind: using GPU/CPU: True - GPU, False - CPU
    :return: None
    """
    if num_cores:
        num_cores = multiprocessing.cpu_count()

    if gpu_ind:
        num_gpu = 1
        num_cpu = 1
    else:
        num_cpu = 1
        num_gpu = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                            device_count={'CPU': num_cpu, 'GPU': num_gpu})
    session = tf.Session(config=config)
    backend.set_session(session)


def resize_images(images, resize_factor):
    """
    resize the images to the same size for later process
    :param images: images - (width, height, depth)
    :param resize_factor: the factor in
    :return: an array of images in the same size
    """
    images_resize_array = []
    for img in images[0, 0:]:
        resize_img = cv2.resize(img, (resize_factor, resize_factor))
        resize_img = np.expand_dims(resize_img, axis=0)
        resize_img = preprocess_input(resize_img)
        images_resize_array.append(resize_img)
    images_resize_array = np.array(images_resize_array)
    images_resize_array = np.rollaxis(images_resize_array, 1, 0)
    images_resize_array = images_resize_array[0]
    return images_resize_array


def train_test_split_first(X, y, k):
    """
    split the data to train and test datasets
    :param X: pictures - (width, height, depth)
    :param y: true labels
    :param k: index to split on
    :return: splited data: X_train, X_test, y_train, y_test
    """
    return np.array(X[:k]), np.array(np.array(X[k:])), np.array(y[:k]), np.array(y[k:])


def prepare_data(params):
    """
    preparing the data - all the preprocessing for prediction
    :param params: the configuration parameters
    :return: pictures ready to be used - (width, length, depth), true labels of classes
    """
    data = scipy.io.loadmat(params["path"])
    images = data['Data']
    labels = data['Labels']
    Y = labels[0]
    resized_images = resize_images(images, params["size"])
    return resized_images, Y


def configure_model(dropout, l1, l2, size):
    """
    configuring the model architecture
    :param dropout: the dropout chance to use
    :param dropout: the dropout chance
    :param l1: the regularization weight for l1 for weight decay
    :param l2: the regularization weight for l2 for weight decay
    :param size: the width and height of a picture
    :return: a configured model
    """
    image_input = Input(shape=(size, size, 3))
    model = VGG16(weights='imagenet', include_top=True, input_tensor=image_input)
    last_layer = model.get_layer('block5_pool').output
    x = Flatten(name='flatten')(last_layer)
    x = Dropout(dropout)(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(dropout)(x)
    out = Dense(params["num_classes"], activation='sigmoid', name='output',
                kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    model = Model(image_input, out)

    # freeze all the layers except the dense layers
    for layer in model.layers[:-3]:
        layer.trainable = False

    return model


def train_model(X_train, y_train, optimizer, batch_size, learning_rate, epochs, dropout, l1, l2, level, context):
    """
    training the model with a specific configuration
    :param X_train: training images
    :param y_train: true labels
    :param optimizer: the optimizer to use for optimization
    :param batch_size: the batch size to learn with
    :param learning_rate: the learning rate of the optimizer
    :param epochs: the number of repetition over the whole data
    :param dropout: the dropout chance
    :param l1: the regularization weight for l1 for weight decay
    :param l2: the regularization weight for l2 for weight decay
    :param level: the level of improvement
    :param context: the context : train/test
    :return: a training history object
    """
    model = configure_model(dropout, l1, l2, params["size"])
    optimizer_m = eval("optimizers.{}(lr={})".format(optimizer, learning_rate))
    tensorboard = TensorBoard(log_dir="logs/{}_{}_{}".format(context, level,
                                                             datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    model.compile(loss='binary_crossentropy', optimizer=optimizer_m,
                  metrics=['accuracy'])
    results_model = model.fit(X_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=params["split_percent"],
                              callbacks=[tensorboard])
    return results_model


def augmentation(X_train, y_train, params):
    """
    doing augmentation - horizontal flip
    :param X_train: training data
    :param y_train: true labels
    :param params: configuration paramaters
    :return: augmented data
    """
    random.seed(params["seed"])
    for i in range(0, params["first_n_split"]):
        if random.random() <= 0.2:
            X_train = np.concatenate((X_train, np.array([X_train[i, :, ::-1, :]])), axis=0)
            y_train = np.concatenate((y_train, np.array([y_train[i]])), axis=0)

    return X_train, y_train


def train_grid_search(X_train, y_train, params, level):
    """
    performs grid search over all the hyper parameters space
    :param X_train: proccesed images (number of pictures, length, width, depth)
    :param y_train: true labels
    :param params: the parameters of the model
    :param level: the level of complexity: basic, improved
    :return: batch_sized from the best configuration, the best model history object
    """
    iterations_results = []
    models = []
    iteration = 1

    for learning_rate in params["hyper_parameters"]["learning_rate"]:
        for batch_size in params["hyper_parameters"]["batch_size"]:
            for optimizer in params["optimizer"]:
                for dropout in params["hyper_parameters"][level]["dropouts"]:
                    for l1 in params["hyper_parameters"][level]["l1"]:
                        for l2 in params["hyper_parameters"][level]["l2"]:
                            for epochs in params["hyper_parameters"]["epochs"]:

                                start_time = time.time()
                                results_model = train_model(X_train, y_train, optimizer, batch_size, learning_rate,
                                                            epochs, dropout, l1, l2, level, "train")
                                print('iteration {}, Training time: {} minutes'.
                                      format(iteration, (time.time() - start_time) / 60))
                                train_loss = results_model.history["loss"][-1]
                                train_acc = results_model.history["acc"][-1]
                                validation_loss = results_model.history["val_loss"][-1]
                                validation_acc = results_model.history["val_acc"][-1]
                                new_row = {"iteration": iteration, "batch_size": batch_size, "optimizer": optimizer,
                                           "learning_rate": learning_rate, "dropout": dropout, "l1": l1, "l2": l2,
                                           "epochs": epochs, "train_loss": train_loss, "train_accuracy": train_acc,
                                           "validation_loss": validation_loss, "validation_acc": validation_acc,
                                           "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                                print(new_row)
                                iterations_results.append(new_row)
                                models.append(results_model)
                                pickle.dump(iterations_results, open("results/backup_{}.p".format(level), 'wb'))
                                iteration += 1

                                # backend.clear_session()
                                # del results_model
                                # configure(gpu_ind=True)
                                # cuda.select_device(0)
                                # cuda.close()

    columns = ["iteration", "batch_size", "optimizer", "learning_rate", "dropout", "l1", "l2", "epochs", "train_loss",
               "train_accuracy", "validation_loss", "validation_acc"]
    df = pd.DataFrame(iterations_results, columns=columns)
    pickle.dump(df, open("results/final_df_train_{}.p".format(level), 'wb'))
    best_index = df[df["validation_acc"] == df["validation_acc"].max()]["iteration"].values[0] - 1
    pickle.dump(models[best_index], open("results/best_model_{}.p".format(level), 'wb'))
    return models[best_index], df.iloc[best_index]["batch_size"]


def alpha(results, level, index_test):
    """
    calculated the alpth probability - first kind of mistake
    :param results: the dataframe of the true valeus and predicted probabilities
    :param level: the level of complexity: basic, improved
    :param index_test: the index that from him the test set starts
    :return: None
    """
    print("the alpha mistake of {} top 5 images are:".format(level))
    worst = results[results["true"] == 1].sort_values(by="predicted", ascending=True).head()
    worst.reset_index(inplace=True)
    worst["index"] = worst["index"].apply(lambda x: x + index_test)
    print(worst)
    pickle.dump(worst, open("results/alpha_{}.p".format(level), "wb"))


def beta(results, level, index_test):
    """
    calculated the beta probability - second kind of mistake
    :param results: the dataframe of the true valeus and predicted probabilities
    :param level: the level of complexity: basic, improved
    :param index_test: the index that from him the test set starts
    :return: None
    """
    print("the beta mistake of {} top 5 images are:".format(level))
    worst = results[results["true"] == 0].sort_values(by="predicted", ascending=False).head()
    worst.reset_index(inplace=True)
    worst["index"] = worst["index"].apply(lambda x: x + index_test + 1)
    print(worst)
    pickle.dump(worst, open("results/beta_alpha_{}.p".format(level), "wb"))


def best_t(precisions, recalls, thresholds):
    """
    calculate the best threshold by F1 measure
    :param precisions: precisions from the precision-recall curve
    :param recalls: recalls from the precision-recall curve
    :param thresholds: thresholds from the precision-recall curve
    :return: the best threshold
    """
    f1 = [2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]) for i in range(0, len(thresholds))]
    return thresholds[np.argmax(f1)]


def precision_recall_curve_calc(y, y_predict_proba, level, accuracy):
    """
    prints the precision recall curve of the model
    :param y: the true class values
    :param y_predict_proba: the probabilities to predict the class
    :param level: the level of complexity: basic, improved
    :param accuracy: accuracy obtained over the test data set
    :return: None
    """
    precision, recall, thresholds = precision_recall_curve(y, y_predict_proba)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('{}: 2-class Precision-Recall curve with score: {}\nthe best threshold is: {}'
              .format(level, accuracy, best_t(precision, recall, thresholds)))
    plt.draw()


def report(best_model, X, y, batch_size, level, index_test):
    """
    gathers all the information over of the performance of the model
    :param best_model: the model that had the best configurion of hyper parameters
    :param X: the test set
    :param y: true labels
    :param batch_size: batch size to calculate each time
    :param level: the level of complexity: basic, improved
    :param index_test: the index that from him the test set starts
    :return: None
    """
    y_predict_proba = best_model.model.predict(X, batch_size=batch_size, verbose=1)
    y_pred = [1 if y > 0.5 else 0 for y in y_predict_proba]
    y_predict_proba = y_predict_proba.reshape(-1)
    loss = log_loss(y, y_predict_proba)
    accuracy = accuracy_score(y, y_pred)
    print("test loss={:.4f}, test accuracy: {:.4f}% of the {} CNN".format(loss, accuracy * 100, level))
    results = pd.DataFrame({"true": y, "predicted": y_predict_proba})
    alpha(results, level, index_test)
    beta(results, level, index_test)
    precision_recall_curve_calc(y, y_predict_proba, level, accuracy)


def get_best_model(X_train, y_train):
    """
    ruunig the program over the improved model over the test data set with best hyper parameters
    found by fine tuning over the validation data set
    :param X_train: images to train the model over
    :param y_train: true labels
    :return: keras history object with the trained model, the batch sized used for training
    """
    df = pd.DataFrame(json.load(open("results/bests.json", "rb"))["best"])
    best_params = df[df["validation_acc"] == df["validation_acc"].max()].reset_index().to_dict()
    results_model = train_model(X_train, y_train, best_params["optimizer"][0], best_params["batch_size"][0],
                                best_params["learning_rate"][0], best_params["epochs"][0], best_params["dropout"][0],
                                best_params["l1"][0], best_params["l2"][0], "improved", "test")

    return results_model, best_params["batch_size"][0]


if __name__ == '__main__':

    start_all_over = False
    levels = ["basic", "improved"]
    configure(gpu_ind=False)
    params = set_params()
    start_time = time.time()
    resized_images, Y = prepare_data(params)
    X_train, X_test, y_train, y_test = train_test_split_first(resized_images, Y, params["first_n_split"])
    if start_all_over:
        for level in levels:
            if level == "improved":
                X_train, y_train = augmentation(X_train, y_train, params)
            best_model, batch_size = train_grid_search(X_train, y_train, params, level)
            report(best_model, X_test, y_test, batch_size, level, params["first_n_split"])
    else:
        X_train, y_train = augmentation(X_train, y_train, params)
        best_model, batch_size = get_best_model(X_train, y_train)
        report(best_model, X_test, y_test, batch_size, "improved", params["first_n_split"])
    print("the total time is: {} minutes".format((time.time() - start_time) / 60))
    plt.show()
