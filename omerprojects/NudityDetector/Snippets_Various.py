#import gc
import numpy as np
import sys
from numpy import save
from numpy import asarray
#from sklearn.model_selection import train_test_split
#from sklearn import model_selection
#from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,multilabel_confusion_matrix


import time
import datetime
import json
import random
import math
import cv2
import pandas as pd
import progressbar
import os
from os import listdir
import matplotlib.pyplot as plt

def imageAugmentation(x_img, y_img, title_List=None, imageShape=(224, 224), maxDegreeToRotate=30, shiftImgBy=0.3,
                      sheerImgBy=0.3, zoomPercent=0.3, flipImages=True, displayExample=True):
    imgH, imgW = imageShape
    x_newList = []
    y_newList = []
    imageTitles = []
    augmentLabel = False

    print("Number of images - Before:  %s" % str(len(x_img)))

    if y_img[0].shape[:2] == imageShape:
        print("augmentLabel - On. Will augment label mask")
        augmentLabel = True

    print('\n')
    bar = progressbar.ProgressBar(maxval=len(x_img),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i, img in enumerate(x_img):
        label = y_img[i]
        originalImg = img
        if imageShape != originalImg.shape[0:2]:
            # If image shape is different than desired shape - Rescale it:
            img = resize_image(originalImg, desired_width=imgW, desired_height=imgH)
            if augmentLabel:
                label = resize_image(label, desired_width=imgW, desired_height=imgH)
        if title_List:
            imageBaseTitle = title_List[i]
        else:
            imageBaseTitle = 'Image %s' % (i)

        x_newList.append(img)
        y_newList.append(label)
        imageTitles.append(imageBaseTitle)

        # Rotate image x degrees:
        if maxDegreeToRotate:
            rotDegree = random.randint(10, maxDegreeToRotate)
            rotatedImage = rotate_img_x_degrees(img, rotDegree)
            if augmentLabel:
                rotatedlabel = rotate_img_x_degrees(label, rotDegree)
                rotatedlabel = np.reshape(rotatedlabel, newshape=(imgH, imgW, 1))
            else:
                rotatedlabel = label
            x_newList.append(rotatedImage)
            y_newList.append(rotatedlabel)
            imageTitle = '%s - Rotated' % (imageBaseTitle)
            imageTitles.append(imageTitle)

        # shift hight and width
        if shiftImgBy:
            randShiftImgBy = random.uniform(0.05, shiftImgBy)
            increaseHight = np.float32([[1, 0, 0], [0, 1, imgH * randShiftImgBy]])
            hightShiftedImage = cv2.warpAffine(img, increaseHight, (imgW, imgH))
            if augmentLabel:
                hightShiftedlabel = cv2.warpAffine(label, increaseHight, (imgW, imgH))
                hightShiftedlabel = np.reshape(hightShiftedlabel, newshape=(imgH, imgW, 1))
            else:
                hightShiftedlabel = label
            x_newList.append(hightShiftedImage)
            y_newList.append(hightShiftedlabel)
            imageTitle = '%s - Shifted H' % (imageBaseTitle)
            imageTitles.append(imageTitle)
            increaseWidth = np.float32([[1, 0, imgW * randShiftImgBy], [0, 1, 0]])
            widthShiftedImage = cv2.warpAffine(img, increaseWidth, (imgW, imgH))
            if augmentLabel:
                widthShiftedlabel = cv2.warpAffine(label, increaseWidth, (imgW, imgH))
                widthShiftedlabel = np.reshape(widthShiftedlabel, newshape=(imgH, imgW, 1))
            else:
                widthShiftedlabel = label
            x_newList.append(widthShiftedImage)
            y_newList.append(widthShiftedlabel)
            imageTitle = '%s - Shifted W' % (imageBaseTitle)
            imageTitles.append(imageTitle)

        # Sheer:
        if sheerImgBy:
            randsheerImgBy = random.uniform(0.05, sheerImgBy)
            sheer = np.float32([[1 - randsheerImgBy, randsheerImgBy, 1], [0, 1, 0]])
            sheeredImage = cv2.warpAffine(img, sheer, (imgW, imgH))
            if augmentLabel:
                sheeredlabel = cv2.warpAffine(label, sheer, (imgW, imgH))
                sheeredlabel = np.reshape(sheeredlabel, newshape=(imgH, imgW, 1))
            else:
                sheeredlabel = label
            x_newList.append(sheeredImage)
            y_newList.append(sheeredlabel)
            imageTitle = '%s - Sheered' % (imageBaseTitle)
            imageTitles.append(imageTitle)

        # Zoom in and out:
        if zoomPercent:
            # Zoom in:
            zoomH = zoomW = int(imgW + (imgW * zoomPercent))
            difference = int((zoomW - imgW) / 2)
            zoomInImage = cv2.resize(src=img, dsize=(zoomH, zoomW), interpolation=cv2.INTER_CUBIC)[
                          difference:imgH + difference, difference:imgW + difference]
            if augmentLabel:
                zoomInlabel = cv2.resize(src=label, dsize=(zoomH, zoomW), interpolation=cv2.INTER_CUBIC)[
                              difference:imgH + difference, difference:imgW + difference]
                zoomInlabel = np.reshape(zoomInlabel, newshape=(imgH, imgW, 1))
            else:
                zoomInlabel = label
            x_newList.append(zoomInImage)
            y_newList.append(zoomInlabel)
            imageTitle = '%s - ZoomIn' % (imageBaseTitle)
            imageTitles.append(imageTitle)

            # Zoom out:
            zoomH = zoomW = int(imgW * (1 - zoomPercent))
            difference = int((imgW - zoomH) / 2)
            zoomOutImage = cv2.resize(src=img, dsize=(zoomH, zoomW), interpolation=cv2.INTER_CUBIC)
            zoomOutImage = cv2.copyMakeBorder(zoomOutImage, difference, difference, difference, difference,
                                              cv2.BORDER_REFLECT)
            # if missing pixels, round it:
            zoomOutImage = cv2.resize(src=zoomOutImage, dsize=(imgW, imgH), interpolation=cv2.INTER_CUBIC)
            if augmentLabel:
                zoomOutlabel = cv2.resize(src=label, dsize=(zoomH, zoomW), interpolation=cv2.INTER_CUBIC)
                zoomOutlabel = cv2.copyMakeBorder(zoomOutlabel, difference, difference, difference, difference,
                                                  cv2.BORDER_REFLECT)
                zoomOutlabel = cv2.resize(src=zoomOutlabel, dsize=(imgW, imgH), interpolation=cv2.INTER_CUBIC)
                zoomOutlabel = np.reshape(zoomOutlabel, newshape=(imgH, imgW, 1))
            else:
                zoomOutlabel = label

            x_newList.append(zoomOutImage)
            y_newList.append(zoomOutlabel)
            imageTitle = '%s - ZoomOut' % (imageBaseTitle)
            imageTitles.append(imageTitle)

            # FlipImages
        if flipImages:
            # flipcode = 0: flip vertically
            # flipcode > 0: flip horizontally
            verticalImage = cv2.flip(img, flipCode=0)
            if augmentLabel:
                verticallabel = cv2.flip(label, flipCode=0)
                verticallabel = np.reshape(verticallabel, newshape=(imgH, imgW, 1))
            else:
                verticallabel = label
            x_newList.append(verticalImage)
            y_newList.append(verticallabel)
            imageTitle = '%s - Vertical Flip' % (imageBaseTitle)
            imageTitles.append(imageTitle)
            horizontalImage = cv2.flip(img, flipCode=1)
            if augmentLabel:
                horizontallabel = cv2.flip(label, flipCode=1)
                horizontallabel = np.reshape(horizontallabel, newshape=(imgH, imgW, 1))
            else:
                horizontallabel = label
            x_newList.append(horizontalImage)
            y_newList.append(horizontallabel)
            imageTitle = '%s - Horizonal Flip' % (imageBaseTitle)
            imageTitles.append(imageTitle)

            if i == 0 and displayExample:
                display_images_in_plot(x_newList, imageTitles)

        bar.update(i)

    if augmentLabel:
        for i, label in enumerate(y_newList):
            # print(label.shape)
            if label.shape[0] != imgH or label.shape[1] != imgW:
                print("Label %s was in an incorrect shape: %s" % (i, str(label.shape)))
                newlabel = resize_image(label, desired_width=imgW, desired_height=imgH)
                x_newList[i] = newlabel

    x_newList = np.asarray(x_newList)
    y_newList = np.asarray(y_newList)
    print("Number of images - After:  %s" % str(len(x_newList)))
    print("Number of label - After:  %s" % str(len(y_newList)))
    return np.asarray(x_newList), np.asarray(y_newList), imageTitles


""" Snippet to quickly display a list of images in a plot """


def display_images_in_plot(list_of_images, list_of_titles=[], show_axis=False, figure_size=(25, 15),
                           images_per_row=None):
    num_of_imgs = len(list_of_images)
    plot_columns = 0
    plot_rows = 0

    if images_per_row:
        plot_columns = images_per_row
        plot_rows = int(num_of_imgs / images_per_row)
    elif num_of_imgs == 1:
        plot_columns = 1
        plot_rows = 1
    elif num_of_imgs == 2:
        plot_columns = 2
        plot_rows = 1
    elif num_of_imgs % 5 == 0:
        plot_columns = 5
        plot_rows = num_of_imgs / 5
    elif num_of_imgs % 4 == 0:
        plot_columns = 4
        plot_rows = num_of_imgs / 4
    elif num_of_imgs % 3 == 0:
        plot_columns = 3
        plot_rows = num_of_imgs / 3
    elif num_of_imgs % 2 == 0:
        plot_columns = 2
        plot_rows = num_of_imgs / 2
    else:
        plot_columns = 5
        plot_rows = math.ceil(num_of_imgs / 5)

    if num_of_imgs > 10:
        figure_size = (plot_columns * 3,  plot_rows * 3)
    fig = plt.figure(figsize=figure_size)

    for i, img in enumerate(list_of_images):
        ax = fig.add_subplot(plot_rows, plot_columns, i + 1)
        if list_of_titles == []:
            ax.set_title('Image number %s' % (i + 1))
        else:
            ax.set_title(list_of_titles[i])
        if not show_axis:
            ax.set_axis_off()
        if len(img.shape) < 3:  # Grayscale
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
    plt.show()

"""Also define a json encoder"""


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


"""Work plan:
Use functions build in hm1:
rotate_img_x_degrees
resize_image

Include all additional image modifications.

When done, build a loop to run through the x and y inputs, and create an augmented new image to each of these parameters
"""


def rotate_img_x_degrees(img, degrees, backgroundcolor=(255, 255, 255), keepOriginalScale=True):
    w, h, _ = img.shape
    img_center = (w / 2, h / 2)  # Center is midway
    # create a set-up for rotation (center of image, degree, ration)
    rotation_setup = cv2.getRotationMatrix2D(img_center, degrees, 1)

    if keepOriginalScale:
        rotated_img = cv2.warpAffine(img, rotation_setup, (w, h))
    else:
        # To adjust the canvas size after the rotation, calculate the new width and height of the rotated image:
        cos = np.abs(rotation_setup[0, 0])
        sin = np.abs(rotation_setup[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        rotation_setup[0, 2] += (nW / 2) - w / 2
        rotation_setup[1, 2] += (nH / 2) - h / 2

        # use the set-up to apply to image (original image, set-up, image size)
        rotated_img = cv2.warpAffine(img, rotation_setup, (nW, nH), borderValue=backgroundcolor)

    return rotated_img


def resize_image(img, desired_width=None, desired_height=None):
    img_height, img_width, _ = img.shape

    if (desired_height is None and desired_width is None):
        print("No scale specified")
    elif (desired_height is not None and desired_width is None):
        image_scale = desired_height / img_height
        new_img_size = (round(img_width * image_scale), round(img_height * image_scale))
    elif (desired_height is None and desired_width is not None):
        image_scale = desired_width / img_width
        new_img_size = (round(img_width * image_scale), round(img_height * image_scale))
    else:  # Scale each part seperatly
        h_scale = desired_height / img_height
        w_scale = desired_width / img_width
        new_img_size = (round(img_width * w_scale), round(img_height * h_scale))

    new_img = cv2.resize(img, new_img_size, interpolation=cv2.INTER_AREA)
    return new_img


""" Snippet to quickly load an image and convert if from BGR to either RGB or Grayscale"""


def load_img_omer(path, isGray=False, showImg=False):
    try:
        img = cv2.imread(path)
    except:
        print("Issues loading image: %s"%(path))
    if img is None:
        print("Could not load image, please check path %s" % path)
    else:
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if showImg:
            fig = plt.figure(figsize=(10, 10))
            plt.title('Loaded image')
            plt.axis('off')
            plt.imshow(img)
    if isGray:
        # if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if showImg:
            fig = plt.figure(figsize=(10, 10))
            plt.title('Loaded image')
            plt.axis('off')
            plt.imshow(img, comap="gray")
    return img

def loadClassDict(ClassDictPath):
  with open(ClassDictPath, 'r') as jsonFile:
    classDict = json.load(jsonFile)

  #Reverse Dict
  reverseClassDict={}
  for key,val in classDict.items():
    reverseClassDict[val]=int(key)

  #Convert ClassDict keys to ints:
  for key,val in reverseClassDict.items():
    classDict[val] = classDict.pop(str(val))

  #Fill in gaps for dict:
  for i in range(max(classDict.keys())):
    if classDict.get(i) is None:
      classDict[i]=None

  print(classDict)
  print(reverseClassDict)
  return classDict,reverseClassDict


def resourcepath(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



def splitDataSet_train_val_test(dataFrame,val_percent=20,test_percent=10):

    #determin train, val, test ration:
    numOfRecords = dataFrame.shape[0]
    num_test = int((numOfRecords*test_percent)/100)
    num_val = int((numOfRecords*val_percent)/100)
    num_train = numOfRecords-num_val-num_test

    print("Total records: %s. Train: %s, Val: %s, Test: %s"%(numOfRecords,num_train,num_val,num_test))

    #Shuffle dataset:
    dataFrame = dataFrame.sample(frac=1)

    dataset_train = dataFrame[0:num_train]
    dataset_val = dataFrame[num_train:num_train+num_val]
    dataset_test = dataFrame[num_train+num_val:]

    return  dataset_train,dataset_val,dataset_test