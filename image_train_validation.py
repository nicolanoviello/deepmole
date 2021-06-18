'''
Il file è stato scaricato da un progetto eseguito su Colaboratory
ed in parte è stato rielaborato per funzionare come script
'''
# Importo le librerie necessarie al processo
import os
import random
import tensorflow as tf
from shutil import copyfile
from shutil import rmtree
from PIL import Image
# Stampo la versione di TensorFlow per verifica
tf.version.VERSION

lesions_dict = {
    'akiec': 'Cheratosi Attinica',
    'bcc': 'Carcinoma Basocellulare',
    'bkl': 'Cheratosi Benigna',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Neo Melanocitico',    
    'vasc': 'Lesione Vascolare'
}

from google.colab import drive
drive.mount('drive')

import pandas as pd
df=pd.read_csv(r"drive/MyDrive/Colab Notebooks/Skin Cancer/HAM10000_metadata.csv")

df

'''
def split_data(SOURCE, TRAINING, TESTING, VALIDATION, TRAIN_SIZE, TEST_SIZE):
    
    # Definisco le dimensioni dei dataset di training e set
    train_set_length = round(len(os.listdir(SOURCE)) * TRAIN_SIZE)
    test_set_length = round(len(os.listdir(SOURCE)) * TEST_SIZE)
    val_set_length = len(os.listdir(SOURCE)) - train_set_length - test_set_length
    
    # Pulisco le directory se già esistono
    try:
        rmtree(TRAINING)
    except OSError as e:
        print("Error: %s : %s" % (TRAINING, e.strerror))
    
    try:
        rmtree(TESTING)
    except OSError as e:
        print("Error: %s : %s" % (TESTING, e.strerror))
    
    try:
        rmtree(VALIDATION)
    except OSError as e:
        print("Error: %s : %s" % (VALIDATION, e.strerror))
    
    # Ricreo le directory
    try:
        os.mkdir(TRAINING)
    except OSError:
        print ("Creation of the directory %s failed" % TRAINING)
    try:
        os.mkdir(TESTING)
    except OSError:
        print ("Creation of the directory %s failed" % TESTING)
    try:
        os.mkdir(VALIDATION)
    except OSError:
        print ("Creation of the directory %s failed" % VALIDATION)
        
    
        
    
    # Creo una lista di url contenenti le immagini
    dataset_new = []
    
    # Controllo se le immagini sono a lunghezza > 0 e li aggiungo nel dataset
    for image_name in os.listdir(SOURCE):
        image_src = SOURCE + image_name
        if(os.path.getsize(image_src) > 0):
            dataset_new.append(image_name)
        else:
            print(image_name + " is zero length, so ignoring")
    
    # Creo un nuovo dataset con url randomizzati
    shuffled_set = random.sample(dataset_new, len(dataset_new))

    # Creo nuove liste di training e set
    train_set = dataset_new[0:train_set_length]
    test_set = dataset_new[train_set_length:(train_set_length+test_set_length)]
    val_set = dataset_new[(train_set_length+test_set_length):]
       
    for image_name in train_set:
        src_train_set = SOURCE +image_name
        dataset_train_set = TRAINING + image_name
        copyfile(src_train_set, dataset_train_set)
    
    for image_name in test_set:
        src_test_set = SOURCE + image_name
        dataset_test_set = TESTING + image_name
        copyfile(src_test_set, dataset_test_set)
    for image_name in val_set:
        src_val_set = SOURCE + image_name
        dataset_val_set = VALIDATION + image_name
        copyfile(src_val_set, dataset_val_set)



AKIEC_SOURCE_DIR = "input/images/akiec/"
BCC_SOURCE_DIR = "input/images/bcc/"
BKL_SOURCE_DIR = "input/images/bkl/"
DF_SOURCE_DIR = "input/images/df/"
MEL_SOURCE_DIR = "input/images/mel/"
NV_SOURCE_DIR = "input/images/nv/"
VASC_SOURCE_DIR = "input/images/vasc/"

TRAINING_AKIEC_DIR = "input/training/akiec/"
TRAINING_BCC_DIR = "input/training/bcc/"
TRAINING_BKL_DIR = "input/training/bkl/"
TRAINING_DF_DIR = "input/training/df/"
TRAINING_MEL_DIR = "input/training/mel/"
TRAINING_NV_DIR = "input/training/nv/"
TRAINING_VASC_DIR = "input/training/vasc/"

TESTING_AKIEC_DIR = "input/testing/akiec/"
TESTING_BCC_DIR = "input/testing/bcc/"
TESTING_BKL_DIR = "input/testing/bkl/"
TESTING_DF_DIR = "input/testing/df/"
TESTING_MEL_DIR = "input/testing/mel/"
TESTING_NV_DIR = "input/testing/nv/"
TESTING_VASC_DIR = "input/testing/vasc/"

VALIDATION_AKIEC_DIR = "input/validation/akiec/"
VALIDATION_BCC_DIR = "input/validation/bcc/"
VALIDATION_BKL_DIR = "input/validation/bkl/"
VALIDATION_DF_DIR = "input/validation/df/"
VALIDATION_MEL_DIR = "input/validation/mel/"
VALIDATION_NV_DIR = "input/validation/nv/"
VALIDATION_VASC_DIR = "input/validation/vasc/"

train_size = .8
test_size = .1
split_data(AKIEC_SOURCE_DIR, TRAINING_AKIEC_DIR, TESTING_AKIEC_DIR, VALIDATION_AKIEC_DIR, train_size, test_size)
split_data(BCC_SOURCE_DIR, TRAINING_BCC_DIR, TESTING_BCC_DIR, VALIDATION_BCC_DIR, train_size, test_size)
split_data(BKL_SOURCE_DIR, TRAINING_BKL_DIR, TESTING_BKL_DIR, VALIDATION_BKL_DIR, train_size, test_size)
split_data(DF_SOURCE_DIR, TRAINING_DF_DIR, TESTING_DF_DIR, VALIDATION_DF_DIR, train_size, test_size)
split_data(MEL_SOURCE_DIR, TRAINING_MEL_DIR, TESTING_MEL_DIR, VALIDATION_MEL_DIR, train_size, test_size)
split_data(NV_SOURCE_DIR, TRAINING_NV_DIR, TESTING_NV_DIR, VALIDATION_NV_DIR, train_size, test_size)
split_data(VASC_SOURCE_DIR, TRAINING_VASC_DIR, TESTING_VASC_DIR, VALIDATION_VASC_DIR, train_size, test_size)
'''

'''
print('TRAINIG DIR')
print(len(os.listdir("input/training/akiec/")))
print(len(os.listdir("input/training/bcc/")))
print(len(os.listdir("input/training/bkl/")))
print(len(os.listdir("input/training/df/")))
print(len(os.listdir("input/training/mel/")))
print(len(os.listdir("input/training/nv/")))
print(len(os.listdir("input/training/vasc/")))
print('TESTING DIR')
print(len(os.listdir("input/testing/akiec/")))
print(len(os.listdir("input/testing/bcc/")))
print(len(os.listdir("input/testing/bkl/")))
print(len(os.listdir("input/testing/df/")))
print(len(os.listdir("input/testing/mel/")))
print(len(os.listdir("input/testing/nv/")))
print(len(os.listdir("input/testing/vasc/")))
print('VALIDATION DIR')
print(len(os.listdir("input/validation/akiec/")))
print(len(os.listdir("input/validation/bcc/")))
print(len(os.listdir("input/validation/bkl/")))
print(len(os.listdir("input/validation/df/")))
print(len(os.listdir("input/validation/mel/")))
print(len(os.listdir("input/validation/nv/")))
print(len(os.listdir("input/validation/vasc/")))
'''

!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/akiec/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/bcc/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/bkl/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/df/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/mel/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/nv/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/vasc/

!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/akiec/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/bcc/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/bkl/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/df/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/mel/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/nv/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/vasc/

!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/akiec/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/bcc/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/bkl/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/df/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/mel/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/nv/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/vasc/

!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/akiec/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/bcc/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/bkl/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/df/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/mel/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/nv/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/validation/vasc/

!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/
!mkdir drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/akiec/
!rm -rf drive/MyDrive/Colab\ Notebooks/Skin\ Cancer/training/akiec/

def split_data(SOURCE, TRAINING, VALIDATION, TRAIN_SIZE):
    
    # Definisco le dimensioni dei dataset di training e set
    train_set_length = round(len(os.listdir(SOURCE)) * TRAIN_SIZE)
    val_set_length = len(os.listdir(SOURCE)) - train_set_length
    
    # Pulisco le directory se già esistono
    # Spostato in un blocco a parte perché su Colab non funziona
    '''
    try:
        rmtree(TRAINING)
    except OSError as e:
        print("Error: %s : %s" % (TRAINING, e.strerror))
    try:
        rmtree(VALIDATION)
    except OSError as e:
        print("Error: %s : %s" % (VALIDATION, e.strerror))
    
    # Ricreo le directory
    try:
        #os.mkdir(TRAINING)
        !mkdir drive/MyDrive/Colab Notebooks/Skin Cancer/training/
    except OSError:
        print ("Creation of the directory %s failed" % TRAINING)
    try:
        #os.mkdir(VALIDATION)
        !mkdir drive/MyDrive/Colab Notebooks/Skin Cancer/validation/
    except OSError:
        print ("Creation of the directory %s failed" % VALIDATION)
        
    '''
        
    
    # Creo una lista di url contenenti le immagini
    dataset_new = []
    
    # Controllo se le immagini sono a lunghezza > 0 e li aggiungo nel dataset
    for image_name in os.listdir(SOURCE):
        image_src = SOURCE + image_name
        if(os.path.getsize(image_src) > 0):
            dataset_new.append(image_name)
        else:
            print(image_name + " is zero length, so ignoring")
    
    # Creo un nuovo dataset con url randomizzati
    shuffled_set = random.sample(dataset_new, len(dataset_new))

    # Creo nuove liste di training e set
    train_set = dataset_new[0:train_set_length]
    val_set = dataset_new[train_set_length:]
       
    for image_name in train_set:
        src_train_set = SOURCE +image_name
        dataset_train_set = TRAINING + image_name
        copyfile(src_train_set, dataset_train_set)
    for image_name in val_set:
        src_val_set = SOURCE + image_name
        dataset_val_set = VALIDATION + image_name
        copyfile(src_val_set, dataset_val_set)

AKIEC_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/akiec/"
BCC_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/bcc/"
BKL_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/bkl/"
DF_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/df/"
MEL_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/mel/"
NV_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/nv/"
VASC_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/vasc/"

TRAINING_AKIEC_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/akiec/"
TRAINING_BCC_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/bcc/"
TRAINING_BKL_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/bkl/"
TRAINING_DF_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/df/"
TRAINING_MEL_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/mel/"
TRAINING_NV_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/nv/"
TRAINING_VASC_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/vasc/"

VALIDATION_AKIEC_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/akiec/"
VALIDATION_BCC_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/bcc/"
VALIDATION_BKL_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/bkl/"
VALIDATION_DF_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/df/"
VALIDATION_MEL_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/mel/"
VALIDATION_NV_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/nv/"
VALIDATION_VASC_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/vasc/"

train_size = .7
split_data(AKIEC_SOURCE_DIR, TRAINING_AKIEC_DIR, VALIDATION_AKIEC_DIR, train_size)
split_data(BCC_SOURCE_DIR, TRAINING_BCC_DIR, VALIDATION_BCC_DIR, train_size)
split_data(BKL_SOURCE_DIR, TRAINING_BKL_DIR, VALIDATION_BKL_DIR, train_size)
split_data(DF_SOURCE_DIR, TRAINING_DF_DIR, VALIDATION_DF_DIR, train_size)
split_data(MEL_SOURCE_DIR, TRAINING_MEL_DIR, VALIDATION_MEL_DIR, train_size)
split_data(NV_SOURCE_DIR, TRAINING_NV_DIR, VALIDATION_NV_DIR, train_size)
split_data(VASC_SOURCE_DIR, TRAINING_VASC_DIR, VALIDATION_VASC_DIR, train_size)

print('TRAINIG DIR')
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/akiec/")))
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/bcc/")))
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/bkl/")))
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/df/")))
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/mel/")))
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/nv/")))
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/vasc/")))
print('VALIDATION DIR')
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/validation/akiec/")))
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/validation/bcc/")))
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/validation/bkl/")))
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/validation/df/")))
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/validation/mel/")))
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/validation/nv/")))
print(len(os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/validation/vasc/")))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

akiec_files = os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/akiec/")
bcc_files = os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/bcc/")
bkl_files = os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/bkl/")
df_files = os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/df/")
mel_files = os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/mel/")
nv_files = os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/nv/")
vasc_files = os.listdir("drive/MyDrive/Colab Notebooks/Skin Cancer/training/vasc/")


pic_index = 2

next_akiec = [os.path.join(TRAINING_AKIEC_DIR, fname) 
                for fname in akiec_files[pic_index-2:pic_index]]
next_bcc = [os.path.join(TRAINING_BCC_DIR, fname) 
                for fname in bcc_files[pic_index-2:pic_index]]



for i, img_path in enumerate(next_akiec+next_bcc):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()

# Valutare aumentation dei dati per bilanciare i dataset

import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

batch_size = 50
TRAINING_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(224,224),
    class_mode='categorical',
  batch_size=batch_size
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(224,224),
    class_mode='categorical',
  batch_size=batch_size
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")
!cp rps.h5 "drive/MyDrive/Colab Notebooks/Skin Cancer/"

!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

# Import the inception model  
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
  # Your Code Here
  layer.trainable = False
  
# Print the model summary
pre_trained_model.summary()

