# -*- coding: utf-8 -*-
"""datagen_model_mobilenet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rZCF8LQ2fpiAJJBknENvUhSMBV0-AGHe
"""

import os
import random
import tensorflow as tf
from shutil import copyfile
from shutil import rmtree
from PIL import Image
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
tf.version.VERSION

from google.colab import drive
drive.mount('drive', force_remount=True)

import numpy as np
TRAINING_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/"
AUGMENTED_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/aug_dir/"
VALIDATION_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/"

num_train_samples = 7135
num_val_samples = 3058
train_batch_size = 10
val_batch_size = 10
image_size = 224

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

train_datagen = ImageDataGenerator(
    preprocessing_function= \
    tf.keras.applications.mobilenet.preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

datagen = ImageDataGenerator(
    preprocessing_function= \
    tf.keras.applications.mobilenet.preprocess_input)


train_batches = train_datagen.flow_from_directory(TRAINING_DIR,
                                            target_size=(image_size,image_size),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(VALIDATION_DIR,
                                            target_size=(image_size,image_size),
                                            batch_size=val_batch_size)

# Note: shuffle=False causes the test dataset to not be shuffled
test_batches = datagen.flow_from_directory(VALIDATION_DIR,
                                            target_size=(image_size,image_size),
                                            batch_size=1,
                                            shuffle=False)

mobile = tf.keras.applications.mobilenet.MobileNet()

# CREATE THE MODEL ARCHITECTURE
from tensorflow.keras.layers import Dense, Dropout

# Exclude the last 5 layers of the above model.
# This will include all layers up to and including global_average_pooling2d_1
x = mobile.layers[-6].output

# Create a new dense layer for predictions
# 7 corresponds to the number of classes
x = Dropout(0.25)(x)
predictions = Dense(7, activation='softmax')(x)

# inputs=mobile.input selects the input layer, outputs=predictions refers to the
# dense layer we created above.

model = Model(inputs=mobile.input, outputs=predictions)

model.summary()

for layer in model.layers[:-23]:
    layer.trainable = False

model.summary()

# Define Top2 and Top3 Accuracy

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', 
              metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

print(valid_batches.class_indices)

class_weights={
    0: 1.0, # akiec
    1: 1.0, # bcc
    2: 1.0, # bkl
    3: 1.0, # df
    4: 3.0, # mel # Try to make the model more sensitive to Melanoma.
    5: 1.0, # nv
    6: 1.0, # vasc
}

from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
filepath = "model_35.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

history = model.fit(train_batches, steps_per_epoch=train_steps, 
                              class_weight=class_weights,
                    validation_data=valid_batches,
                    validation_steps=val_steps,
                    epochs=35, verbose=1,
                   callbacks=callbacks_list)

model.save("model_35.h5")
!cp model.h5 "drive/MyDrive/Colab Notebooks/Skin Cancer/"
