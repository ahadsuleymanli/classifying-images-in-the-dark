import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import load_model
from utils.nets import LeNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

from utils.helpers import read_files, flat_intensities, v_histogram, h_histogram, image_sized_down
from utils.helpers import get_extra_tests


MODEL_DIR = "saved_models/lenet.hd5"

def get_saved_model(name=None):
    return load_model(MODEL_DIR)

def get_prediction(model,image):
    _image = cv2.resize(image, (28,28))
    _image = _image/255.0
    _image = np.expand_dims(_image, axis=0)
    prediction = model.predict([_image],)
    acc = prediction[0][0]
    print(acc)
    print(prediction[0][1])
    result = "day" if prediction[0][1] > 0.5 else "night"
    result_string = "Accuracy:{:.4f}, {}".format(acc,result)
    return result_string


if __name__ == "__main__":
    # set the matplotlib backend so figures can be saved in the background
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type=int, default=20, required=False)
    ap.add_argument("-p", "--plot", type=str, default="plot.png",help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())

    EPOCHS = args["epochs"]
    INIT_LR = 1e-3
    BS = 32

    # scale the raw pixel intensities to the range [0, 1]
    image_array, label_array = read_files(image_sized_down)
    image_array = image_array/255.0
    # replace [day,night] classes with [1,0] numeric variants 
    label_array = [1 if x == "day" else 0 for x in label_array]
    label_array = np.array(label_array)

    # train test split
    (trainX, testX, trainY, testY) = train_test_split(image_array, label_array, test_size=0.25, random_state=42)
    
    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

    model = LeNet.build(width=28, height=28, depth=3, classes=2)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1)

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
    image_array, label_array = get_extra_tests(image_sized_down)
    image_array = image_array/255.0
    # replace [day,night] classes with [1,0] numeric variants 
    label_array = [1 if x == "day" else 0 for x in label_array]
    label_array = np.array(label_array)
    y_test = to_categorical(label_array, num_classes=2) 
    x_test = image_array
    print("Evaluate on extra_tests data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)