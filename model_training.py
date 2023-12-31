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
import tensorflow as tf


INIT_LR = 1e-4
EPOCHS =20
BS =32

DATASET_DIRECTORY = r"/home/daniel/Documents/PythonProjects/Mask_Recognition/mask_recognition/dataset"
CLASSIFICATION_CATEGORIES = ["with_mask","without_mask"]

print("Loading images...")


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
print("Loaded")
labels = LabelBinarizer().fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX,testX,trainY,testY) =  train_test_split(data,labels,test_size=0.20,stratify=labels,random_state=42)


image_changer = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

baseModel = MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128,activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2,activation="softmax")(headModel)


finalModel = Model(inputs=baseModel.input,outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("Compiling model...")
adam_optimizer = tf.keras.optimizers.legacy.Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
finalModel.compile(loss="binary_crossentropy",optimizer=adam_optimizer,metrics=["accuracy"])

print("Training head...")
trained_head = finalModel.fit(
                image_changer.flow(trainX,trainY,batch_size=BS),
                steps_per_epoch=len(trainX)//BS,
                validation_data=(testX,testY),
                validation_steps=len(testX)//BS,
                epochs=EPOCHS)
print("Evaluating network...")
preditions = finalModel.predict(testX,batch_size=BS)
preditions = np.argmax(preditions,axis=1)

#print(classification_report(testY.argmax(axis=1),preditions,target_names=LabelBinarizer().classes_))

print("Saving model")

finalModel.save("mask_detector.model",save_format="h5")


#print("Plotting...")
#N = EPOCHS
#plt.style.use("ggplt")
#plt.figure()
#plt.plot(np.arrange(0,N),trained_head.history["loss"],label="train_loss")
#plt.plot(np.arrange(0,N),trained_head.history["val_loss"],label="val_loss")
#plt.plot(np.arrange(0,N),trained_head.history["accuracy"],label="train_acc")
#plt.plot(np.arrange(0,N),trained_head.history["val_accuracy"],label="val_acc")
#plt.title("Loss and Accuracy")
#plt.xlabel("Epoch")
#plt.ylabel("Loss/Accuracy")
#plt.legend(loc="lower left")
#plt.savefig("plot.png")


