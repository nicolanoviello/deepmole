'''
Il file è stato scaricato da un progetto eseguito su Colaboratory
ed in parte è stato rielaborato per funzionare come script
'''
# Importo le librerie necessarie
import tensorflow as tf
from tensorflow.keras import Model

'''
# Monto la directory di Google Drive
from google.colab import drive
drive.mount('drive')
'''
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
# Definisco le funzioni custom e carico il modello
def accuratezza_top_2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def accuratezza_top_3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

new_model = tf.keras.models.load_model('drive/MyDrive/Colab Notebooks/Skin Cancer/model.h5', 
                                               custom_objects={'categorical_accuracy': categorical_accuracy,'top_2_accuracy': accuratezza_top_2,'top_3_accuracy': accuratezza_top_3}, 
                                               compile=True)
# Definisco il dizionario con le varie tipologie di neo
import heapq
lesions_dict = {
    0: 'MAL - Cheratosi Attinica',
    1: 'MAL - Carcinoma Basocellulare',
    2: 'BEN - Cheratosi Benigna',
    3: 'BEN - Dermatofibroma',
    4: 'MAL - Melanoma',
    5: 'BEN - Neo Melanocitico',    
    6: 'BEN - Lesione Vascolare'
}
# Creo una funzione in grado di mostrare il risultato dell'inferenza
# però arricchito con la percentuale di accuratezza
# inoltre per evitare allarmismi viene mostrato anche il secondo risultato 
# con la relativa perncentuale di accuratezza
def risultato_analisi(classes):
    ris_list = list(classes[0])
    ris_max = ris_list.index(max(ris_list))
    print("Dall'analisi risulta un possibile {0} con valore {1}".format(lesions_dict[ris_max],max(ris_list)))
    posizioni = heapq.nlargest(3, range(len(ris_list)), key=ris_list.__getitem__)
    print("Il secondo valore più probabile è invece {0} con valore {1}".format(lesions_dict[posizioni[1]],ris_list[posizioni[1]]))

# Per l'inferenza in questa fase è stato scelto l'utilizzo di Colaboratory
# tramite il caricamento diretto dell'immagine

import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = new_model.predict(images, batch_size=10)
  risultato_analisi(classes)

