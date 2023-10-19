from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os



INIT_LR = 1e-4
EPOCHS =20
BS =32

DATASET_DIRECTORY = r"/home/daniel/Documents/PythonProjects/Mask_Recognition/mask_recognition/dataset"
CLASSIFICATION_CATEGORIES = ["with_mask","without_mask"]

print("Loading ...")


data=[]
labels=[]

for category in CLASSIFICATION_CATEGORIES:
    category_path = os.path.join(DATASET_DIRECTORY,category)
    for item in os.listdir(category_path):
        image_path = os.path.join(category_path,item)
        image = load_img(image_path,target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

print("loaded")




