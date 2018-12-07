import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Reshape
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorboard
import time
from keras.callbacks import TensorBoard

NAME = 'logo_classification_{}'.format(int(time.time()))
tb = TensorBoard(log_dir='logs/{}'.format(NAME))

def CNN(traindata, trainlabel, testdata, testlabel):
    model = Sequential()

    model.add(BatchNormalization(axis=1, input_shape=(1, 32, 32)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #fully connected layer
    model.add(Flatten())
    model.add(Dense(500, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11, init='normal', activation='softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu': 0})))
    model.fit(x=traindata, y=trainlabel, batch_size=64, nb_epoch=5, shuffle=True, verbose=1, validation_split=0.1, callbacks=[tb])
    scores = model.evaluate(x=testdata, y=testlabel, verbose=1, batch_size=32)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    model.save('logo_classify_augdata.h5')
    # plot_model(model, to_file='model.png', show_shapes=True)

def amount(path):
    dir = os.listdir(path)
    tol = 0
    for each in dir:
        f = os.listdir(path + each + '/')
        for ff in f:
            tol += 1
    return tol

def genelabel(path, tol):
    data = np.empty((tol, 1, 32, 32), dtype='float32')
    label = np.empty((tol, ), dtype='uint8')
    n = 0
    dir = os.listdir(path)
    for each in dir:
        f = os.listdir(path + each + '/')
        for ff in f:
            img = cv2.imread(path + each + '/' + ff, 0) #0=灰度图
            # print(img.shape)
            # print(img.shape)
            # img_contents = tf.read_file(path + each + '/' + ff)
            # img = tf.image.decode_jpeg(img, channels=3)
            if img is None: continue
            img = cv2.resize(img, (32, 32))
            # img = np.transpose(img, (2, 0, 1))
            array = np.asarray(img, dtype='float32')
            array = array / 255
            data[n, :, :, :] = array
            label[n] = int(each)
            n += 1
    return  data, label



CLASSES = 11
trainpath = 'F:/vehicle_detection/data/train_aug2/'
testpath = 'F:/vehicle_detection/data/test_aug2/'
traintol = amount(trainpath) #1000s
testtol = amount(testpath) #500s
traindata, trainlabel = genelabel(trainpath, traintol)
testdata, testlabel = genelabel(testpath, testtol)
trainlabel = np_utils.to_categorical(trainlabel, CLASSES)
testlabel = np_utils.to_categorical(testlabel, CLASSES)
CNN(traindata, trainlabel, testdata, testlabel)






