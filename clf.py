import sqlite3

import numpy as np
import pandas as pd

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd

import os
dir_black = os.path.join('Data/cloudy')
dir_Cinder = os.path.join('Data/desert')
dir_Laterite = os.path.join('Data/green_area')
dir_peat = os.path.join('Data/Soil')
dir_yellow = os.path.join('Data/water')

dir_yellow2 = os.path.join('Data/Fire')
dir_yellow1 = os.path.join('Data/Buildings')
import tensorflow as tf
from tensorflow import keras
image_size = 224
batch_size = 16


target_size = (image_size, image_size)
input_shape = (image_size, image_size, 3)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMAGE_SHAPE = (224, 224)
train_datagen = ImageDataGenerator(rescale=1. / 255.,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        validation_split=0.15)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255., validation_split=0.15)
valid_generator = valid_datagen.flow_from_directory(
   'Data/',
    subset="validation",
    batch_size = batch_size,
    target_size=IMAGE_SHAPE
)

train_generator = train_datagen.flow_from_directory(
        'Data/',
        target_size=(224, 224),
        batch_size = batch_size)
model = tf.keras.models.Sequential([

    # The first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(7, activation='softmax')
])
model.summary()
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
from keras.metrics import Precision , Recall

model.compile(
  optimizer=optimizer,
  loss='categorical_crossentropy',
  metrics=['accuracy',Precision(),Recall()])
# total_sample = train_generator.n
# n_epochs = 36
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.hdf5', monitor = 'val_accuracy', verbose = 1, save_best_only = True)
history = model.fit(
        train_generator,
        epochs = 5,
        validation_data= valid_generator,
        callbacks = [checkpoint])
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
model.save('my_model.h5')
model.save(filepath="save_model/")
# from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
# pre = Precision()
# re = Recall()
# acc = BinaryAccuracy()
# for X, y in valid_generator:

#     yhat = model.predict(X)
#     pre.update_state(y, yhat)
#     re.update_state(y, yhat)
#     acc.update_state(y, yhat)
from tensorflow.keras.models import load_model

data= load_model('best_model.hdf5')
class_names = sorted(os.listdir('Data'))
class_names
['cloudy', 'desert', 'green_area', 'Soil', 'water','Fire','Buildings']


def eval_model(model):
    class_names = ['cloudy', 'desert', 'green_area', 'Soil', 'water','Fire','Buildings']
    val_gen = valid_datagen.flow_from_directory(
        directory='Data/',
        target_size=(224, 224),
        shuffle=False,
        classes=class_names,  # classes to predict
        seed=43,  # to make the result reproducible
        subset="validation")

    model_preds = model.predict(val_gen)
    predicted_labels = np.argmax(model_preds, axis=1)
    classes = val_gen.classes

    # list of target class names
    print(classification_report(predicted_labels, classes))

    cm = confusion_matrix(classes, predicted_labels)

    figure(figsize=(20, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
    plt.title('Confusion matrix', fontsize=15)
    plt.colorbar()
    plt.xticks(range(7), class_names, fontsize=12, rotation=60)
    plt.yticks(range(7), class_names, fontsize=12, verticalalignment="center")
    # Iteating over cells to write number of prediction for each class
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > np.max(cm) / 2. else "black")
    plt.xlabel('Predicted label', fontsize=15)
    plt.ylabel('True label', fontsize=15)


eval_model(data)
preds = sorted(os.listdir('Data'))
preds_idx = [0,1,2,3,4,5,6]
preds = dict(zip(preds_idx,preds))

conn = sqlite3.connect('Form.db')
with conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM imgsave")
    rows = cursor.fetchall()
    for row in rows:
        filename = row[0]

img = Image.open(filename)

# Resize the image to the model's input size (228x228)
img = img.resize((224, 224))

# Convert the image to a numpy array and normalize the pixel values
img_array = np.array(img) / 255.0

# Add a batch dimension to the array
img_array = np.expand_dims(img_array, axis=0)

# Make the prediction
pred = model.predict(img_array)

# Get the true label
true_label = np.argmax(pred)

# Print the true label name
print('True Label:', class_names[true_label])

# Show the image and the prediction
plt.axis("off")
plt.imshow(img)
plt.title(class_names[true_label])
print('Prediction:', pred)
from tkinter import messagebox
Data=class_names[true_label]
print(Data)
if Data=="cloudy":
    messagebox.showinfo("Sat","Cloudy Area")

if Data=="desert":
  messagebox.showinfo("Sat","Desert Area")

if Data=="green_area":
   messagebox.showinfo("Sat","Green Area")
if Data=="Soil":
    messagebox.showinfo("Sat","Soil Area")
if Data=="water":
    messagebox.showinfo("Sat","Water Area")
if Data=="Fire":
    messagebox.showinfo("Sat","Fire Area")
if Data=="Buildings":
    messagebox.showinfo("Sat","Buildings")