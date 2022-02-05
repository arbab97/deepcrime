
from __future__ import print_function
import keras, sys
from operators import activation_function_operators
from operators import training_data_operators
from operators import bias_operators
from operators import weights_operators
from operators import optimiser_operators
from operators import dropout_operators,hyperparams_operators
from operators import training_process_operators
from operators import loss_operators
from utils import mutation_utils
from utils import properties
from keras import optimizers
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

def main(model_name):
    model_location = os.path.join('trained_models', model_name)
    custom_data_path = '/media/rabi/Data/11111/Task 99/deepcrime/datasets/custom_data.csv'
    data = pd.read_csv(custom_data_path)
    (data, test) = train_test_split(data, test_size=0.2)
    data = data.drop(columns=['id'])
    labels = data['diagnosis']
    data.drop(columns=['diagnosis'], inplace=True)
    data = data.iloc[:, 0:29]
    data_x = data
    n = Normalizer()
    data_x = n.fit_transform(data_x)
    map = {'M': 1, 'B': 0}
    labels = labels.map(map)
    test_y = test['diagnosis']
    test = test.drop(columns=['id', 'diagnosis'])
    test = test.iloc[:, 0:29]
    test = n.transform(test)
    test_y = test_y.map(map)
    test_y.head()
    if (not os.path.exists(model_location)):
        epochs = 300
        batch_size = 32
        model = Sequential()
        model.add(Dense(12, input_dim=29, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
        model.fit(data_x, labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test, test_y))
        model.save(os.path.join('trained_models', 'custom_trained.h5'))
        score = model.evaluate(data_x, labels, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score
    else:
        graph1 = tf.Graph()
        with graph1.as_default():
            session1 = tf.compat.v1.Session()
            with session1.as_default():
                model = tf.keras.models.load_model(model_location)
                score = model.evaluate(data_x, labels, verbose=0)
                print(('score:' + str(score)))
        return score
if (__name__ == '__main__'):
    score = main('')
