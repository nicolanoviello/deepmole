'''
Il file è stato scaricato da un progetto eseguito su Colaboratory
ed in parte è stato rielaborato per funzionare come script
'''
# Importo le librerie necessarie
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
# Verifico la versione di tensorflow
tf.version.VERSION

'''
# Monto la directory di Google Drive
from google.colab import drive
drive.mount('drive')
'''
# Definisco i path delle dir di training e validation
import numpy as np
TRAINING_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/"
VALIDATION_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/"

# Imposto i valori per eseguire gli scriot di datagen
num_train_samples = 7135
num_val_samples = 3058
train_batch_size = 10
val_batch_size = 10
image_size = 224

# Calcolo gli step 
train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

# Creo una funzione di data generator 
# aumentando le immagini del dataset
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
# Creo un datagen specifico per MobileNet
# senza augmentation per i test
datagen = ImageDataGenerator(
    preprocessing_function= \
    tf.keras.applications.mobilenet.preprocess_input)

# Creo i bach di training e validation
train_batches = train_datagen.flow_from_directory(TRAINING_DIR,
                                            target_size=(image_size,image_size),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(VALIDATION_DIR,
                                            target_size=(image_size,image_size),
                                            batch_size=val_batch_size)
'''
# Creo i bach di test
# Note: shuffle=False perché il dataset di test non deve essere randomizzato
# Tensorflow 2.0 usa validation sia per loss che per metrics
test_batches = datagen.flow_from_directory(VALIDATION_DIR,
                                            target_size=(image_size,image_size),
                                            batch_size=1,
                                            shuffle=False)
'''

'''
Definizione del modello
'''                                       

model = tf.keras.applications.mobilenet.MobileNet()

# CREDO L'ARCHITETTURA DEL MODELLO
from tensorflow.keras.layers import Dense, Dropout
# Cambio il livello della predizione finale
# In base al numero delle classi ( 7 classi)
model = Dropout(0.25)(model)
predictions = Dense(7, activation='softmax')(model)

# inputs=mobile.input imposta il layer di input 
# outputs=predictions il layer di output
# appena creato e modellato sul nostro classificatore

model = Model(inputs=model.input, outputs=predictions)

model.summary()
# Blocco il training sugli ultimi 25 livelli
model.layers[:-25].trainable = False

model.summary()
'''
OUTPUT 
# Stampo solo ultimi layer del modello
conv_pw_13 (Conv2D)          (None, 7, 7, 1024)        1048576   
_________________________________________________________________
conv_pw_13_bn (BatchNormaliz (None, 7, 7, 1024)        4096      
_________________________________________________________________
conv_pw_13_relu (ReLU)       (None, 7, 7, 1024)        0         
_________________________________________________________________
global_average_pooling2d (Gl (None, 1024)              0         
_________________________________________________________________
dropout (Dropout)            (None, 1024)              0         
_________________________________________________________________
dense (Dense)                (None, 7)                 7175      
=================================================================
Total params: 3,236,039
Trainable params: 1,869,831
Non-trainable params: 1,366,208
_________________________________________________________________
'''
# Qui ho seguito un'impelementazione esistente
# Ho solo regolato i pesi in base al dataset

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def accuratezza_top_2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)
def accuratezza_top_3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

# Scelta dei parametri
model.compile(Adam(lr=0.01), loss='categorical_crossentropy', 
              metrics=[categorical_accuracy, accuratezza_top_2, accuratezza_top_3])

print(valid_batches.class_indices)
'''
OUTPUT
{'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
'''
class_weights={
    0: 2.0, # 'akiec': 'Cheratosi Attinica'
    1: 2.0, # 'bcc': 'Carcinoma Basocellulare'
    2: 1.0, # 'bkl': 'Cheratosi Benigna'
    3: 1.0, # 'df': 'Dermatofibroma'
    4: 3.0, # 'mel': 'Melanoma'
    5: 1.0, # 'nv': 'Neo Melanocitico'
    6: 1.0, # 'vasc': 'Lesione Vascolare'
}
# Definizione dei callback
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
filepath = "model_mobinet.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00005)
                              
# Definisco la lista dei callback da passare al modello                            
callbacks_list = [checkpoint, reduce_lr]
# Effettuo il fit del modello
history = model.fit(train_batches, steps_per_epoch=train_steps, 
                              class_weight=class_weights,
                    validation_data=valid_batches,
                    validation_steps=val_steps,
                    epochs=35, verbose=1,
                   callbacks=callbacks_list)
# Salvo la migliore versione del modello
model.save("model_mobinet.h5")
'''
# Passo il modello dalla memoria ad un file fisico in drive
!cp model_final.h5 "drive/MyDrive/Colab Notebooks/Skin Cancer/"
'''

