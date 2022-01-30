import csv
import os
import numpy as np
from PIL import Image
# import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# from keras.utils import to_categorical

reader = csv.reader(open("file1.csv"))
headers = next(reader, None)
# print(reader)
Y = []
X = []
print("Currently I am here")
for row in reader:
    # print(int(row[1]))
    # print("hi")
    image = Image.open(
            "COVID-19_Radiography_Dataset/{}".format(row[0]))
    # print(image)
    image2arr = np.array(image)
    # print(image2arr)
    X.append(image2arr)
    Y.append(int(row[1]))
print("Now here")
X = np.array(X)
Y = np.array(Y)
print(len(Y))
print(X)

print(X.shape)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',
          kernel_initializer='he_uniform', input_shape=(384, 512, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3),
          activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3),
          activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3),
          activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3),
          activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3),
          activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3),
          activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))

model.add(Flatten())
# model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dense(4, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
model.summary()
print(tf.__version__)
model.fit(X, Y, 10, epochs=40)
