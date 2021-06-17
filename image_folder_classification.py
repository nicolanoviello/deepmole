# Il file è stato scaricato da un progetto eseguito su Colaboratory
# ed in parte è stato rielaborato per funzionare come script
import tensorflow as tf
import os

'''
# Controllo la versione di Tensorflow
tf.version.VERSION
'''

'''
# Monto la directory di Google Drive
from google.colab import drive
drive.mount('drive')
# Scompatto il file zip con le immagini
!unzip -uq "drive/MyDrive/Colab Notebooks/Skin Cancer/images.zip" -d "drive/MyDrive/Colab Notebooks/Skin Cancer/images/"
'''

# Definisco la directory con le immagini
image_dir = os.path.join('drive/MyDrive/Colab Notebooks/Skin Cancer/images/')
# Verifico il numero totale di immagini
print('total images:', len(os.listdir(image_dir)))
'''
OUTPUT
total images: 10193
'''
# Importo CSV con metatata e li inserisco in una struttura Pandas DataFrame
import pandas as pd
df=pd.read_csv(r"drive/MyDrive/Colab Notebooks/Skin Cancer/HAM10000_metadata.csv")
df
'''
OUTPUT
lesion_id	image_id	dx	dx_type	age	sex	localization
0	HAM_0000118	ISIC_0027419	bkl	histo	80.0	male	scalp
1	HAM_0000118	ISIC_0025030	bkl	histo	80.0	male	scalp
2	HAM_0002730	ISIC_0026769	bkl	histo	80.0	male	scalp
3	HAM_0002730	ISIC_0025661	bkl	histo	80.0	male	scalp
4	HAM_0001466	ISIC_0031633	bkl	histo	75.0	male	ear
...	...	...	...	...	...	...	...
10010	HAM_0002867	ISIC_0033084	akiec	histo	40.0	male	abdomen
10011	HAM_0002867	ISIC_0033550	akiec	histo	40.0	male	abdomen
10012	HAM_0002867	ISIC_0033536	akiec	histo	40.0	male	abdomen
10013	HAM_0000239	ISIC_0032854	akiec	histo	80.0	male	face
10014	HAM_0003521	ISIC_0032258	mel	histo	70.0	female	back
10015 rows × 7 columns
'''
# Costriusco le URL delle immagini partendo dai nomi
# ed inserisco le informazioni nel dataframe
df['image_name'] = df['image_id'].astype(str)+'.jpg'
df
'''
	lesion_id	image_id	dx	dx_type	age	sex	localization	image_name
0	HAM_0000118	ISIC_0027419	bkl	histo	80.0	male	scalp	ISIC_0027419.jpg
1	HAM_0000118	ISIC_0025030	bkl	histo	80.0	male	scalp	ISIC_0025030.jpg
2	HAM_0002730	ISIC_0026769	bkl	histo	80.0	male	scalp	ISIC_0026769.jpg
3	HAM_0002730	ISIC_0025661	bkl	histo	80.0	male	scalp	ISIC_0025661.jpg
4	HAM_0001466	ISIC_0031633	bkl	histo	75.0	male	ear	ISIC_0031633.jpg
...	...	...	...	...	...	...	...	...
10010	HAM_0002867	ISIC_0033084	akiec	histo	40.0	male	abdomen	ISIC_0033084.jpg
10011	HAM_0002867	ISIC_0033550	akiec	histo	40.0	male	abdomen	ISIC_0033550.jpg
10012	HAM_0002867	ISIC_0033536	akiec	histo	40.0	male	abdomen	ISIC_0033536.jpg
10013	HAM_0000239	ISIC_0032854	akiec	histo	80.0	male	face	ISIC_0032854.jpg
10014	HAM_0003521	ISIC_0032258	mel	histo	70.0	female	back	ISIC_0032258.jpg
10015 rows × 8 columns
'''
# Definisco i path per le varie categorie del dataset
from shutil import move
all_images = os.listdir(image_dir)
AKIEC_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/akiec/"
BCC_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/bcc/"
BKL_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/bkl/"
DF_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/df/"
MEL_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/mel/"
NV_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/nv/"
VASC_SOURCE_DIR = "drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/vasc/"

# Creo le directory
try:
    os.mkdir(AKIEC_SOURCE_DIR)
except OSError:
    print ("Creation of the directory %s failed" % AKIEC_SOURCE_DIR)
try:
    os.mkdir(BCC_SOURCE_DIR)
except OSError:
    print ("Creation of the directory %s failed" % BCC_SOURCE_DIR)
try:
    os.mkdir(BKL_SOURCE_DIR)
except OSError:
    print ("Creation of the directory %s failed" % BKL_SOURCE_DIR)
try:
    os.mkdir(DF_SOURCE_DIR)
except OSError:
    print ("Creation of the directory %s failed" % DF_SOURCE_DIR)
try:
    os.mkdir(MEL_SOURCE_DIR)
except OSError:
    print ("Creation of the directory %s failed" % MEL_SOURCE_DIR)
try:
    os.mkdir(NV_SOURCE_DIR)
except OSError:
    print ("Creation of the directory %s failed" % NV_SOURCE_DIR)
try:
    os.mkdir(VASC_SOURCE_DIR)
except OSError:
    print ("Creation of the directory %s failed" % VASC_SOURCE_DIR)

# Sposto tutte le immagini in una cartella relativa alla specifica categoria
co = 0
for image in all_images:
    # Controllo a quale classe appartiene l'immagine 
    category = df[df['image_name'] == image]['dx']
    # Prendo solo il nome della classe
    try:
        categoria = str(list(category)[0])
    except IndexError:
        print('errore')
        #continue
    path_from = os.path.join('drive/MyDrive/Colab Notebooks/Skin Cancer/images/', image)
    path_to = os.path.join('drive/MyDrive/Colab Notebooks/Skin Cancer/images_final/', categoria, image)
    try :
        move(path_from, path_to)
    except FileNotFoundError:
        #print ('File Not Found')
        continue
    co += 1
print('Moved {} images.'.format(co))
'''
Il risultato finale è una directory "images_final" contenente tutte le immagini organizzate in
sottodirectory con il nome della classe di appartenenza
'''
