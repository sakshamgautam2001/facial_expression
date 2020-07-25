# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 05:48:46 2020

@author: DELL
"""

######Importing Necessary directories##########
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from livelossplot.tf_keras import PlotLossesCallback
from IPython.display import SVG, Image
import tensorflow as tf
print("Tensorflow version:", tf.__version__)

######Training and test set generators#####
size=48
batch=64
img_data_generator=ImageDataGenerator(horizontal_flip=True)

training_gen=img_data_generator.flow_from_directory("train",
                                                    target_size=(size,size),
                                                    color_mode="grayscale",
                                                    batch_size=batch,
                                                    class_mode='categorical',
                                                    shuffle=True)
test_gen=img_data_generator.flow_from_directory("test",
                                                target_size=(size,size),
                                                color_mode="grayscale",
                                                batch_size=batch,
                                                class_mode='categorical',
                                                shuffle=False)


#############Creation of CNN model begins here###############
cnnmodel=Sequential()

#Layer-1
cnnmodel.add(Conv2D(64,(3,3), padding='same', input_shape=(48,48,1)))
cnnmodel.add(BatchNormalization())
cnnmodel.add(Activation('relu'))
cnnmodel.add(MaxPooling2D(pool_size=(2,2)))
cnnmodel.add(Dropout(0.25))

#Layer-2
cnnmodel.add(Conv2D(128,(5,5), padding='same'))
cnnmodel.add(BatchNormalization())
cnnmodel.add(Activation('relu'))
cnnmodel.add(MaxPooling2D(pool_size=(2,2)))
cnnmodel.add(Dropout(0.25))

#Layer-3
cnnmodel.add(Conv2D(256,(3,3), padding='same' ))
cnnmodel.add(BatchNormalization())
cnnmodel.add(Activation('relu'))
cnnmodel.add(MaxPooling2D(pool_size=(2,2)))
cnnmodel.add(Dropout(0.25))


cnnmodel.add(Flatten())

#Layer Dense - 1
cnnmodel.add(Dense(256))
cnnmodel.add(BatchNormalization())
cnnmodel.add(Activation('relu'))
cnnmodel.add(Dropout(0.25))

#Layer Dense - 2
cnnmodel.add(Dense(512))
cnnmodel.add(BatchNormalization())
cnnmodel.add(Activation('relu'))
cnnmodel.add(Dropout(0.25))


cnnmodel.add(Dense(7, activation='softmax'))

optimizer=Adam(lr=0.0005)
cnnmodel.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
cnnmodel.summary()

##############Fitting the model################
epochs=15
steps_epoch=training_gen.n//training_gen.batch_size
valid_steps=test_gen.n//test_gen.batch_size
reducelr=ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                           patience=2, min_lr=0.00001, mode='auto')
check_point = ModelCheckpoint("model_weights.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)
callbacks = [PlotLossesCallback(), check_point, reducelr]

hist = cnnmodel.fit(
    x=training_gen,
    steps_per_epoch=steps_epoch,
    epochs=epochs,
    validation_data = test_gen,
    validation_steps = valid_steps,
    callbacks=callbacks
)




cnnmodel_json = cnnmodel.to_json()
with open("model.json", "w") as json_file:
    json_file.write(cnnmodel_json)


