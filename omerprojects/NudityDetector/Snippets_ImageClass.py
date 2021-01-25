from Snippets_Various import load_img_omer,resize_image,JsonEncoder,display_images_in_plot,rotate_img_x_degrees
import sys
import gc
import numpy as np

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



#import tensorflow as tsf
# Version 1.1.1
class OmerImageClass():

    def __init__(self, imageId, reverseClassDict,imagePath="", imageNP=None,
                 saveFolder='/content/drive/My Drive/Harvard HW/Final Project/DataSet',printResults=0):
        self.imageId = imageId
        self.imageRelativePath=""
        self.imagePath = imagePath.replace('\\','/')
        if printResults: print("File path is: "+self.imagePath)
        if self.imagePath.find("Classes/") != -1:
            self.imageRelativePath = self.imagePath.split("Classes/", 1)
            if printResults: print("Splitted path: " + str(self.imageRelativePath))
            self.imageRelativePath=self.imageRelativePath[1]
        self.imageNP = imageNP
        self.originalImg = None
        self.reverseClassDict = reverseClassDict
        self.saveFolder = saveFolder
        self.labelCategory = np.zeros(shape=(1, len(self.reverseClassDict)), dtype=float)
        if self.imagePath == "" and imageNP is None:
            print("No path or image provided to Class.")
        elif self.imagePath == "" and imageNP is not None:
            self.originalImg = imageNP
        elif self.imagePath != "":
            if os.path.exists(imagePath):
                head_tail = os.path.split(self.imageRelativePath)
                self.fullImageName = head_tail[1]
                self.cleanImageName = self.fullImageName.split('.')[0]
                self.allImageClasses = head_tail[0].split(r'/')
                if printResults: print(self.allImageClasses)
                self.mainClassName = self.allImageClasses[0]
                try:
                    self.mainClassLabel = int(self.reverseClassDict[self.mainClassName])
                except:
                    print(self.cleanImageName)
                    print(self.allImageClasses)
                    print(self.imagePath)
                    print("Wasn't able to match value for: %s" % self.mainClassName)

                modifiedImageNameString = ""
                for i, classN in enumerate(self.allImageClasses):
                    label = int(self.reverseClassDict[classN])
                    # print('%s - %s'%(classN,label))
                    self.labelCategory[0][label] = 1.00
                    if i == 0:
                        modifiedImageNameString = classN
                    else:
                        modifiedImageNameString += "_%s" % (classN)
                self.modifiedImageName = "Image %s Class %s" % (self.imageId, modifiedImageNameString)
                self.savePath = os.path.join(self.saveFolder, self.modifiedImageName + '.jpg')
                # print(self.modifiedImageName)
                # print(self.labelCategory)
                if self.imageNP is None:
                    self.originalImg = load_img_omer(self.imagePath)
                    if self.originalImg is None:
                        print("Imag %s could not be loaded. Will be deleted - Path: %s" % (
                        self.cleanImageName, self.imagePath))
                        os.remove(self.imagePath)
                else:
                    self.originalImg = self.imageNP
                    self.reshapedImage = self.reshape_Img(normelize=False)
            else:
                print("Image path does not exists: %s" % imagePath)
        else:
            print("Somthing went wrong:")
            print(imageId)
            print(imagePath)
            print(imageNP)

    def reshape_Img(self, imageShape=(224, 224), normelize=True, keepOriginal=True):
        self.reshapedImage = resize_image(self.originalImg, desired_height=imageShape[0], desired_width=imageShape[1])
        if normelize:
            self.normelizedImage = self.reshapedImage / 255
            if keepOriginal == False:
                self.originalImg == self.normelizedImage
            return self.normelizedImage
        else:
            return self.reshapedImage

    def save_reshaped_img(self, overrideExisting=False):
        self.savePath = os.path.join(self.saveFolder, self.modifiedImageName + '.jpg')
        if os.path.exists(self.savePath) == False or overrideExisting == True:
            print("Savibg image to: %s" % (self.savePath))
            cv2.imwrite(self.savePath, cv2.cvtColor(self.reshapedImage, cv2.COLOR_RGB2BGR))
        else:
            print("File %s already exists" % (self.imagePath))





def augment_img_Class(imgClass, maxDegreeToRotate=30, shiftImgBy=0.3, sheerImgBy=0.3, zoomPercent=0.3, flipImages=True,
                      displayExample=False):
    listOfAugmentedIMGs = list()

    img = imgClass.reshapedImage
    imgW, imgH = img.shape[:2]
    # Rotate image x degrees:
    if maxDegreeToRotate:
        rotDegree = random.randint(10, maxDegreeToRotate)
        rotatedImage = rotate_img_x_degrees(img, rotDegree)
        imageTitle = '%s_Rotated' % (imgClass.modifiedImageName)
        Newclass = OmerImageClass(imgClass.imageId, imageNP=rotatedImage, imagePath=imgClass.imagePath,
                                    reverseClassDict=imgClass.reverseClassDict, saveFolder=imgClass.saveFolder)
        Newclass.modifiedImageName = imageTitle
        listOfAugmentedIMGs.append(Newclass)

    # shift hight and width
    if shiftImgBy:
        randShiftImgBy = random.uniform(0.05, shiftImgBy)
        increaseHight = np.float32([[1, 0, 0], [0, 1, imgH * randShiftImgBy]])
        hightShiftedImage = cv2.warpAffine(img, increaseHight, (imgW, imgH))
        imageTitle = '%s_Shifted_High' % (imgClass.modifiedImageName)
        Newclass = OmerImageClass(imgClass.imageId, imageNP=hightShiftedImage, imagePath=imgClass.imagePath,
                                    reverseClassDict=imgClass.reverseClassDict, saveFolder=imgClass.saveFolder)
        Newclass.modifiedImageName = imageTitle
        listOfAugmentedIMGs.append(Newclass)

        increaseWidth = np.float32([[1, 0, imgW * randShiftImgBy], [0, 1, 0]])
        widthShiftedImage = cv2.warpAffine(img, increaseWidth, (imgW, imgH))
        imageTitle = '%s_Shifted_Wide' % (imgClass.modifiedImageName)
        Newclass = OmerImageClass(imgClass.imageId, imageNP=widthShiftedImage, imagePath=imgClass.imagePath,
                                    reverseClassDict=imgClass.reverseClassDict, saveFolder=imgClass.saveFolder)
        Newclass.modifiedImageName = imageTitle
        listOfAugmentedIMGs.append(Newclass)

    # Sheer:
    if sheerImgBy:
        randsheerImgBy = random.uniform(0.05, sheerImgBy)
        sheer = np.float32([[1 - randsheerImgBy, randsheerImgBy, 1], [0, 1, 0]])
        sheeredImage = cv2.warpAffine(img, sheer, (imgW, imgH))
        imageTitle = '%s_Sheered' % (imgClass.modifiedImageName)
        Newclass = OmerImageClass(imgClass.imageId, imageNP=sheeredImage, imagePath=imgClass.imagePath,
                                    reverseClassDict=imgClass.reverseClassDict, saveFolder=imgClass.saveFolder)
        Newclass.modifiedImageName = imageTitle
        listOfAugmentedIMGs.append(Newclass)

    # Zoom in and out:
    if zoomPercent:
        # Zoom in:
        zoomH = zoomW = int(imgW + (imgW * zoomPercent))
        difference = int((zoomW - imgW) / 2)
        zoomInImage = cv2.resize(src=img, dsize=(zoomH, zoomW), interpolation=cv2.INTER_CUBIC)[
                      difference:imgH + difference, difference:imgW + difference]
        imageTitle = '%s_ZoomIn' % (imgClass.modifiedImageName)
        Newclass = OmerImageClass(imgClass.imageId, imageNP=zoomInImage, imagePath=imgClass.imagePath,
                                    reverseClassDict=imgClass.reverseClassDict, saveFolder=imgClass.saveFolder)
        Newclass.modifiedImageName = imageTitle
        listOfAugmentedIMGs.append(Newclass)

        # Zoom out:
        zoomH = zoomW = int(imgW * (1 - zoomPercent))
        difference = int((imgW - zoomH) / 2)
        zoomOutImage = cv2.resize(src=img, dsize=(zoomH, zoomW), interpolation=cv2.INTER_CUBIC)
        zoomOutImage = cv2.copyMakeBorder(zoomOutImage, difference, difference, difference, difference,
                                          cv2.BORDER_REFLECT)
        # if missing pixels, round it:
        zoomOutImage = cv2.resize(src=zoomOutImage, dsize=(imgW, imgH), interpolation=cv2.INTER_CUBIC)
        imageTitle = '%s_ZoomOut' % (imgClass.modifiedImageName)
        Newclass = OmerImageClass(imgClass.imageId, imageNP=zoomOutImage, imagePath=imgClass.imagePath,
                                    reverseClassDict=imgClass.reverseClassDict, saveFolder=imgClass.saveFolder)
        Newclass.modifiedImageName = imageTitle
        listOfAugmentedIMGs.append(Newclass)

    if flipImages:
        # flipcode = 0: flip vertically
        # flipcode > 0: flip horizontally
        verticalImage = cv2.flip(img, flipCode=0)
        zoomOutImage = cv2.resize(src=zoomOutImage, dsize=(imgW, imgH), interpolation=cv2.INTER_CUBIC)
        imageTitle = '%s_VerticalFlip' % (imgClass.modifiedImageName)
        Newclass = OmerImageClass(imgClass.imageId, imageNP=verticalImage, imagePath=imgClass.imagePath,
                                    reverseClassDict=imgClass.reverseClassDict, saveFolder=imgClass.saveFolder)
        Newclass.modifiedImageName = imageTitle
        listOfAugmentedIMGs.append(Newclass)

        horizontalImage = cv2.flip(img, flipCode=1)
        imageTitle = '%s_HorizonalFlip' % (imgClass.modifiedImageName)
        Newclass = OmerImageClass(imgClass.imageId, imageNP=horizontalImage, imagePath=imgClass.imagePath,
                                    reverseClassDict=imgClass.reverseClassDict, saveFolder=imgClass.saveFolder)
        Newclass.modifiedImageName = imageTitle
        listOfAugmentedIMGs.append(Newclass)

        if displayExample:
            print([x.modifiedImageName for x in listOfAugmentedIMGs])
            display_images_in_plot([x.originalImg for x in listOfAugmentedIMGs],
                                   [x.modifiedImageName for x in listOfAugmentedIMGs])

        return listOfAugmentedIMGs



""" Create function to modify images
If label is mask, augment it the same way the image is augmented
"""

if __name__ == '__main__':

    print('Class Loaded successfully')
    reverseClassDict={'Penis': 0, 'Vagina': 1, 'Butt': 2, 'BreastWoman': 3, 'BreastMan': 4, 'BathingSuite': 5, 'Banana': 6, 'Peach': 7}
    aclass = OmerImageClass(imageId=1,reverseClassDict=reverseClassDict, imagePath='D:/Google Drive/Harvard HW/Final Project/Classes/Peach/Banana/Peach_Banana - 2.jpg',saveFolder='D:\Google Drive\Harvard HW\Final Project\DataSet',printResults=1)
    print(aclass.imagePath)
    aclass.reshape_Img()
    aclass.save_reshaped_img(overrideExisting=True)