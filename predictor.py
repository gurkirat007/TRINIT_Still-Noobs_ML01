import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


def predict(image_name):
    model = tf.keras.models.load_model("cxr.h5")

    image = Image.open(image_name)
    image2arr = np.array(image)
    image2arr = cv2.cvtColor(image2arr, cv2.COLOR_GRAY2BGR)
    image2arr = image2arr/255
    # image2arr = tf.constant(image2arr)
    print(image2arr.shape)
    # converted = tf.image.grayscale_to_rgb(image2arr)
    # print(converted.shape)
    # print(image2arr.shape)
    image2arr = image2arr.reshape(1, 299, 299, 3)
    x = np.argmax(model.predict(image2arr))
    # print(type(x))
    # print(x)
    # switch
    if x == 0:
        return("you might Covid")
    elif x == 1:
        return("you might lung opacity")
    elif x == 2:
        return("you aren't having any disease")
    elif x == 3:
        return("you might have viral pneumonia")


# np.argmax(model.predict(image2arr)
if __name__ == '__main__':
    print(predict("Normal-2783.png"))
