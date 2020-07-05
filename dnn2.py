import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import load_model
from utils.nets_v2 import LeNet_v2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import math

from utils.helpers import read_files, flat_intensities, v_histogram, just_image
from utils.helpers import get_extra_tests


MODEL_DIR = "saved_models/lenet_v2.hd5"

def get_saved_model(name=None):
    return load_model(MODEL_DIR)

def get_prediction(model,image):
    _image = cv2.resize(image, (28,28))
    histogram = v_histogram(image)
    _image = _image/255.0
    _image = np.expand_dims(_image, axis=0)
    histogram = np.expand_dims(histogram, axis=0)
    prediction = model.predict([_image,histogram],)
    acc = max(prediction[0][0],prediction[0][1])
    acc = math.floor(acc*100)/100
    result = "day" if prediction[0][1] > 0.5 else "night"
    result_string = "Accuracy:{:.2f}, {}".format(acc,result)
    return result_string


if __name__ == "__main__":
    # set the matplotlib backend so figures can be saved in the background
    matplotlib.use("Agg")
    random.seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type=int, default=24, required=False)
    ap.add_argument("-p", "--plot", type=str, default="plot.png",help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())

    EPOCHS = args["epochs"]
    INIT_LR = 1e-3
    BS = 32

    # get image and label arrays from disk
    image_array, label_array = read_files(just_image)
    histogram_array = np.array([v_histogram(x) for x in image_array])
    image_array = np.array([cv2.resize(x, (28,28)) for x in image_array])

    # scale the raw pixel intensities to the range [0, 1]
    image_array = image_array/255.0
    # replace [day,night] classes with [1,0] numeric variants 
    label_array = [1 if x == "day" else 0 for x in label_array]
    label_array = np.array(label_array)

    # convert the labels from integers to vectors
    label_array = to_categorical(label_array, num_classes=2)

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

    def gen_flow_for_two_inputs(X1, X2, y):
        genX1 = aug.flow(X1,y,  batch_size=BS,seed=666)
        genX2 = aug.flow(X1,X2, batch_size=BS,seed=666)
        while True:
                X1i = genX1.next()
                X2i = genX2.next()
                #Assert arrays are equal - this was for peace of mind, but slows down training
                #np.testing.assert_array_equal(X1i[0],X2i[0])
                yield [X1i[0], X2i[1]], X1i[1]
    

    model = LeNet_v2.build(width=28, height=28, depth=3, feature_shape=histogram_array[0].shape, classes=2)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


    # train test split
    image_array,image_array_v,histogram_array,histogram_array_v,label_array,label_array_v = train_test_split(image_array,histogram_array,label_array, test_size=0.15, random_state=666)

    # train the network
    print("[INFO] training network...")
    # H = model.fit({"x_in":image_array,"y_in":histogram_array},{"day_night":label_array}, validation_split=0.3, epochs=EPOCHS, verbose=1)
    H = model.fit(gen_flow_for_two_inputs(image_array,histogram_array,label_array), validation_data=([image_array_v,histogram_array_v],label_array_v),steps_per_epoch=len(image_array)//BS, epochs=EPOCHS, verbose=1)


    # save the model to disk
    print("[INFO] serializing network...")
    model.save(MODEL_DIR)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

    plt.title("Training Loss and Accuracy on Day/Night")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


    # testing images from the extra tests folder
    image_array, label_array = get_extra_tests(just_image)
    histogram_array = np.array([v_histogram(x) for x in image_array])
    image_array = np.array([cv2.resize(x, (28,28)) for x in image_array])
    image_array = image_array/255.0
    # replace [day,night] classes with [1,0] numeric variants 
    label_array = [1 if x == "day" else 0 for x in label_array]
    label_array = np.array(label_array)
    label_array = to_categorical(label_array, num_classes=2) 
    print("Evaluate on extra_tests data")
    results = model.evaluate({"x_in":image_array,"y_in":histogram_array}, label_array, batch_size=128)
    print("test loss, test acc:", results)