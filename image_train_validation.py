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

# Creo un dizionario con le varie tipologie di nei definite dal dataset
lesions_dict = {
    'akiec': 'Cheratosi Attinica',
    'bcc': 'Carcinoma Basocellulare',
    'bkl': 'Cheratosi Benigna',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Neo Melanocitico',    
    'vasc': 'Lesione Vascolare'
}

'''
# Monto la directory di Google Drive
from google.colab import drive
drive.mount('drive')
'''

# Definisco un metodo per la divisione delle immagini già organizzate
# nelle diverse directory in directory di TRAINING / VALIDATION

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
            print(image_name + " il file è di dimensione 0, probabilmente l'immagine è corrotta")
    
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

# Definisco le dir di input
AKIEC_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/akiec/"
BCC_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/bcc/"
BKL_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/bkl/"
DF_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/df/"
MEL_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/mel/"
NV_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/nv/"
VASC_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/vasc/"
# Definisco le dir di training
TRAINING_AKIEC_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/akiec/"
TRAINING_BCC_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/bcc/"
TRAINING_BKL_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/bkl/"
TRAINING_DF_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/df/"
TRAINING_MEL_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/mel/"
TRAINING_NV_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/nv/"
TRAINING_VASC_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/training/vasc/"
# Definisco le dir di validation
VALIDATION_AKIEC_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/akiec/"
VALIDATION_BCC_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/bcc/"
VALIDATION_BKL_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/bkl/"
VALIDATION_DF_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/df/"
VALIDATION_MEL_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/mel/"
VALIDATION_NV_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/nv/"
VALIDATION_VASC_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/validation/vasc/"
# Setto la percentuale della dimensione di training ed eseguo
# il metodo su ogni categoria
train_size = .7
split_data(AKIEC_SOURCE_DIR, TRAINING_AKIEC_DIR, VALIDATION_AKIEC_DIR, train_size)
split_data(BCC_SOURCE_DIR, TRAINING_BCC_DIR, VALIDATION_BCC_DIR, train_size)
split_data(BKL_SOURCE_DIR, TRAINING_BKL_DIR, VALIDATION_BKL_DIR, train_size)
split_data(DF_SOURCE_DIR, TRAINING_DF_DIR, VALIDATION_DF_DIR, train_size)
split_data(MEL_SOURCE_DIR, TRAINING_MEL_DIR, VALIDATION_MEL_DIR, train_size)
split_data(NV_SOURCE_DIR, TRAINING_NV_DIR, VALIDATION_NV_DIR, train_size)
split_data(VASC_SOURCE_DIR, TRAINING_VASC_DIR, VALIDATION_VASC_DIR, train_size)

# Stampo in numero degli elementi di ogni directory

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

'''
OUTPUT

TRAINIG DIR
230
369
782
83
790
4778
103
VALIDATION DIR
99
158
335
35
339
2048
44
'''


'''
# Commento questa parte in quanto è servita solo
# per valutare l'effettiva lettura delle immagini
# andando a visionare qualche immagine presa random nelle directory

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
'''

# Valutare aumentation dei dati per bilanciare i dataset


'''
# Seguono diversi tentativi relativi alla generazione dei datagen
# e all'implementazione di algoritmi custom
# oltre che relativi all'utilizzo di modelli tipo Inception V3
# tutti con risultati non ottimali, quindi abbandonati

import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
# Generazione dati
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
# Creazione modello
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
# Salvataggio modello
model.save("rps.h5")
!cp rps.h5 "drive/MyDrive/Colab Notebooks/Skin Cancer/"
# Download Inception v3
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

# Import del modello  
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Creato un'istanza del modello Inception V3 con i pesi pre-trained
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

# Imposto i layers pre-trained del modelo non-trainabili
for layer in pre_trained_model.layers:
  layer.trainable = False
  
# Stampo il summary del modello
pre_trained_model.summary()

'''