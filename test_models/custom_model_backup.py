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
import tensorflow.keras as k
import tensorflow as tf
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split


data = pd.read_csv('/media/rabi/Data/11111/Task 99/deepcrime/datasets/custom_data.csv')
data.head()
data.isnull().sum()

data,test = train_test_split(data,test_size = 0.2)
data,validation = train_test_split(data,test_size = 0.2)
data.shape

data = data.drop(columns = ['id'])
data['diagnosis'].unique()

labels = data['diagnosis']
data.drop(columns = ['diagnosis'],inplace = True)
data = data.iloc[:,0:29]
data.head()

daig = validation['diagnosis']
validation.drop(columns = ['id','diagnosis'],inplace=True)
validation.iloc[:,0:29]

validation = validation.iloc[:,0:29]

data_x = data
data_x.shape


 
n = Normalizer()
data_x = n.fit_transform(data_x)

# labels = tf.keras.utils.to_categorical(labels)
map = {'M':1,'B':0}
labels.value_counts()

labels = labels.map(map)
labels

validation = n.transform(validation)
daig = daig.map(map)
test.head()

test_y = test['diagnosis']
test = test.drop(columns = ['id','diagnosis'])
test= test.iloc[:,0:29]
test.head()


test = n.transform(test)
test_y = test_y.map(map)
test_y.head()
 

epoch = 600
model = k.models.Sequential()
model.add( k.layers.Dense(12,input_dim =29,activation = 'relu'))

model.add(k.layers.Dropout(0.5))
model.add(k.layers.Dense(5,activation = 'relu'))
model.add(k.layers.Dropout(0.5))
#                                     k.layers.Dense(5,activation = 'relu'),
#                                     k.layers.Dropout(0.5),
model.add(k.layers.Dense(1,activation = 'sigmoid'))

model_check = k.callbacks.ModelCheckpoint('model_check.h5',save_best_only=True)
model.compile(loss = ['binary_crossentropy'],optimizer =keras.optimizers.Adam(learning_rate = 0.001),metrics = ['accuracy'])
model.fit(data_x,labels,epochs=epoch,verbose=1,validation_data = (validation,daig), batch_size = 32)
score = model.evaluate(validation,daig, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
 