"""
Author: Ahad Suleymanli

plotting histograms for each channel in HSV 
in order to visualise the prevalence in difference in them between night and day images
"""

import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import path

IMAGE_DIR = path.join(path.dirname(path.realpath(__file__)),os.pardir,'day_night_images')
DAY_DIR = path.join(IMAGE_DIR,'day')
NIGHT_DIR = path.join(IMAGE_DIR,'night')


def get_hist(img,channel):
    hist = cv2.calcHist([img],[channel],None,[64],[0,256])
    hist = hist/sum(hist)
    return hist

for subdir, dirs, files in os.walk(DAY_DIR):
    img = cv2.imread(DAY_DIR + os.sep + files[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    break
for subdir, dirs, files in os.walk(NIGHT_DIR):
    img2 = cv2.imread(NIGHT_DIR + os.sep + files[1])
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    break
    
grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

grayscale_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)


f, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(16,8))
ax1.imshow(img,cmap='gray'); ax1.set_title('Day')
ax2.plot(get_hist(hsv_img,0)); ax2.set_title('H')
ax3.plot(get_hist(hsv_img,1)); ax3.set_title('S')
ax4.plot(get_hist(hsv_img,2)); ax4.set_title('V')
ax5.plot(get_hist(grayscale_img,0)); ax5.set_title('Grayscale Intensity')
ax6.imshow(img2,cmap='gray'); ax6.set_title('Night')
ax7.plot(get_hist(hsv_img2,0)); ax7.set_title('H')
ax8.plot(get_hist(hsv_img2,1)); ax8.set_title('S')
ax9.plot(get_hist(hsv_img2,2)); ax9.set_title('V')
ax10.plot(get_hist(grayscale_img2,0)); ax10.set_title('Grayscale Intensity')
plt.show()

