import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import time

img = cv2.imread('test.jpg', 1);

#function that displays initial image
def display(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#########################################################

#function that displays an image in gray scale
def grayScale(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', imgGray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#########################################################

#function displays image through blue channel
def blueChannel(img):
    imgCopy = img.copy()
    red = imgCopy[:, :, 2]
    green = imgCopy[:, :, 1]
    blue = imgCopy[:, :, 0]
    red[0:] = 0
    green[0:] = 0
    imgCopy = cv2.merge((blue, green, red))
    cv2.imshow('image', imgCopy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#########################################################

#function that applys the gaussian filter to a gray scale image
def gaussianFilter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    display(blur)

#########################################################

#function that rotates the initial image 90 degrees
def rotateImg(img):
    (cols, rows) = img.shape[:2]
    center = (rows / 2, cols / 2)
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated90 = cv2.warpAffine(img, M, (rows, cols))
    display(rotated90)

#########################################################

#fnction for resizing image
def resizeImg(img):
    (cols, rows) = img.shape[:2]
    res = cv2.resize(img, (int(rows / 2), cols), interpolation = cv2.INTER_CUBIC)
    display(res)

#########################################################

#function for edge detection
def edgeDetection(img):
    edges = cv2.Canny(img, 100, 200)
    display(edges)

#########################################################

#function for image segmentation
def imgSegmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (ret, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    display(thresh)

#########################################################

#function for face detection
def faceDetection(img):
    imgCopy = img.copy()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgGray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = imgGray[y:y + h, x:x + w]
        roi_color = imgCopy[y:y + h, x:x + w]
    display(imgCopy)

#########################################################

#function for fram display
def frameDisplay():

    cap = cv2.VideoCapture('test.avi')

    i = 0
    while (i < 15):
        (success, image) = cap.read()
        cv2.imshow('capture', image)
        i = i + 1
        cv2.waitKey(500)
    cap.release()
    cv2.destroyAllWindows()

#########################################################
print('press alt + F4 to show the next frame:')
display(img)
grayScale(img)
blueChannel(img)
gaussianFilter(img)
rotateImg(img)
resizeImg(img)
edgeDetection(img)
imgSegmentation(img)
faceDetection(img)
frameDisplay()