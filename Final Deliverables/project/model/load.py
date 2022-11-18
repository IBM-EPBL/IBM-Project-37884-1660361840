import numpy as np
import keras.models
from PIL import Image
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def init():
    num_classes = 10
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    model = Sequential()

    #First Layer
    model.add(Conv2D(6, (5, 5), padding='same', activation='relu',strides=(1, 1), input_shape=((28,28,1))))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    #Second Layer
    model.add(Conv2D(16, (5, 5), padding='valid',strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    #Forth Layer
    model.add(Dense(120, activation='relu'))
    #Fifth Layer
    model.add(Dense(84, activation='relu'))
    #Output Layer
    model.add(Dense(num_classes, activation='softmax'))

 
    
    #load weights into new model
    model.load_weights("weights2.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adadelta(),
                metrics=['accuracy'])
    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)

    return model, None