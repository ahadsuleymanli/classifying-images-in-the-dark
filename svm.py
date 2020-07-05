"""
Author: Ahad Suleymanli

"""

import sys
import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection  import train_test_split
import time
import pickle
import argparse as ap

from utils.helpers import read_files, flat_intensities, v_histogram, h_histogram
from utils.helpers import add_extra_tests
MODEL_DIR = "saved_models/svm_model.pkl"


def get_saved_svm(name=None):
    print("Loading SVM saved model...")
    svm = pickle.load(open(MODEL_DIR, "rb"))
    print("Loaded")
    return svm

def get_prediction(svm,image):
    x = h_histogram(image)
    x = x.reshape(1, -1)
    return svm.predict(x)[0]
    

if __name__ == "__main__":
    argparser = ap.ArgumentParser(description="Trains the SVM model")
    argparser.add_argument("--train", action='store_true')
    args = argparser.parse_args()

    if not os.path.isfile(MODEL_DIR) and not args.image:
        print("No model found, try: python svm.py --train")
        exit()

    feature_array, label_array = read_files(h_histogram)
    X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.05, random_state=42)

    X_test,y_test = add_extra_tests(h_histogram,X_test,y_test)

    if argparser.parse_args(['--train']):
        print("Fitting")

        # Fitting model
        svm = SVC()
        svm.fit(X_train, y_train)

        print("Saving model...")
        pickle.dump(svm, open(MODEL_DIR, "wb"))
    else:
        svm = get_saved_svm()
    

    print("Testing...\n")
    right = 0
    total = 0
    for x, y in zip(X_test, y_test):
        x = x.reshape(1, -1)
        prediction = svm.predict(x)[0]
        if y == prediction:
            right += 1
        total += 1

    accuracy = float(right)/float(total)*100
    print ("{} accuracy".format(accuracy))
