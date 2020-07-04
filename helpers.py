"""
Helper functions for file reading and feature extraction

"""
import sys
import os
import cv2
import numpy as np
from os import path

IMAGE_DIR = path.join(path.dirname(path.realpath(__file__)),'day_night_images')
CLASS_DIRS = {
    "day": path.join(IMAGE_DIR,'day'), 
    "night": path.join(IMAGE_DIR,'night')
}

FLAT_INTENSITIES_SHAPE = (32,32)
HISTOGRAM_BINS = 64


def flat_intensities(image_file):
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imread(image_file)
    img = cv2.resize(img, FLAT_INTENSITIES_SHAPE, interpolation = cv2.INTER_CUBIC)
    img = img.flatten()
    img = img/np.mean(img)
    return img

def _histogram(image_file,channel):
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img],[channel],None,[HISTOGRAM_BINS],[0,256])
    hist = hist/sum(hist)
    return hist

def v_histogram(image_file):
    '''
        gets the histogram of value channel
    '''
    return _histogram(image_file , 2)

def h_histogram(image_file):
    '''
        gets the histogram of hue channel
    '''
    return _histogram(image_file , 0)

def just_image(image_file):
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_files(feature_function):
    '''
        feature_function: the function that reads the image and extracts a feature
        the feature is (N0,1) shape
    '''
    print ("Reading files...")
    feature_list = []
    label_list   = []
    for class_name, class_dir in CLASS_DIRS.items():
        for subdir, dirs, files in os.walk(class_dir):
            for image in files:
                label_list.append(class_name)
                feature_list.append(feature_function(class_dir + os.sep + image))
                
    return np.asarray(feature_list), np.asarray(label_list)



if __name__ == "__main__":
    imgs, labels = read_files(feature_function=h_histogram)
    print(imgs.shape)
