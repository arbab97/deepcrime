from __future__ import print_function
import keras, sys
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
    custom_data_path='/media/rabi/Data/11111/Task 99/deepcrime/datasets/custom_data.csv'
    data = pd.read_csv(custom_data_path)
    # data.head()
    # data.isnull().sum()

    data,test = train_test_split(data,test_size = 0.2)
    #data,validation = train_test_split(data,test_size = 0.2)
    # data.shape

    data = data.drop(columns = ['id'])
    #data['diagnosis'].unique()   #comment this

    labels = data['diagnosis']
    data.drop(columns = ['diagnosis'],inplace = True)
    data = data.iloc[:,0:29]
    #data.head() 

    # daig = validation['diagnosis']
    # validation.drop(columns = ['id','diagnosis'],inplace=True)
    # validation.iloc[:,0:29]

    # validation = validation.iloc[:,0:29]

    data_x = data
    # data_x.shape

    n = Normalizer()
    data_x = n.fit_transform(data_x)

    # labels = tf.keras.utils.to_categorical(labels)
    map = {'M':1,'B':0}
    # labels.value_counts()

    labels = labels.map(map)
    # labels

    # validation = n.transform(validation)
    # daig = daig.map(map)
    # test.head()

    test_y = test['diagnosis']
    test = test.drop(columns = ['id','diagnosis'])
    test= test.iloc[:,0:29]
    # test.head()


    test = n.transform(test)
    test_y = test_y.map(map)
    test_y.head()
    
    if (not os.path.exists(model_location)):
        epochs = 300   # Initial was 600
        batch_size = 32 # Default of mnist was 128
        model = Sequential()
        model.add( Dense(12,input_dim =29,activation = 'relu'))  #Warning: The mnist one uses input_shape !
        model.add(Dropout(0.5))
        model.add(Dense(5,activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation = 'sigmoid'))

        model.compile(loss = keras.losses.binary_crossentropy,optimizer =keras.optimizers.Adam(learning_rate = 0.001),metrics = ['accuracy'])
        # New Mapping: x_train->data_x; y_train->labels; x_test-> test; y_test->test_y
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

if __name__ == '__main__':
    score = main('')



# NEXT:  
# Remove input_dim if required