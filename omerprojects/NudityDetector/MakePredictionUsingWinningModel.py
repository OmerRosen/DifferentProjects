from omerprojects.NudityDetector.ImageClassifierClass import OmerSuperModel
from omerprojects.__init__ import app
# from torchvision import transforms as transforms

import gc
import numpy as np
from numpy import save
from numpy import asarray
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, multilabel_confusion_matrix

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

from omerprojects.NudityDetector.Snippets_Various import load_img_omer, resize_image, JsonEncoder, loadClassDict, display_images_in_plot
from omerprojects.NudityDetector.Snippets_ImageClass import OmerImageClass, augment_img_Class
from omerprojects.NudityDetector.Snippets_SuperModelClass import OmerSuperModel
import gc
import os
import numpy as np
from numpy import save
from numpy import asarray
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, multilabel_confusion_matrix
import json
import tensorflow as tf

from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def loadAndPrepareImgForPrediction(img_path, imgShape=(256, 256, 3)):
    img = load_img_omer(path=img_path)
    imgName = os.path.basename(img_path)
    imgTitle = os.path.splitext(imgName)[0]

    if img is None:
        print("Could not load image: %s" % (img_path))
    else:
        desired_height = imgShape[0]
        desired_width = imgShape[1]

        img = resize_image(img, desired_height=desired_height, desired_width=desired_width)

    return img, imgTitle


def loadClassDict(ClassDictPath):
    with open(ClassDictPath, 'r') as jsonFile:
        classDict = json.load(jsonFile)

    # Reverse Dict
    reverseClassDict = {}
    for key, val in classDict.items():
        reverseClassDict[val] = int(key)

    # Convert ClassDict keys to ints:
    for key, val in reverseClassDict.items():
        classDict[val] = classDict.pop(str(val))

    # Fill in gaps for dict:
    for i in range(max(classDict.keys())):
        if classDict.get(i) is None:
            classDict[i] = None

    #print(classDict)
    #print(reverseClassDict)
    return classDict, reverseClassDict


def takeImagePath_ReturnPredictions(imagesPathList, requestedModelAbsPath, classDictionaryPath=os.path.join(app.config['BASE_FOLDER'], 'NudityDetector/Classes/ClassDictionary.json')):
    listOfImageFiles = imagesPathList

    basePath = os.path.join(app.config['BASE_FOLDER'], 'NudityDetector')
    winningModelPath = requestedModelAbsPath

    classDict, reverseClassDict = loadClassDict(ClassDictPath=classDictionaryPath)

    if not os.path.exists(winningModelPath):
        print("Could not find model at: %s" % winningModelPath)
        raise ("Could not find model at: %s" % winningModelPath)
    else:
        winningModel = load_model(filepath=winningModelPath, custom_objects={'OmerSuperModel': OmerSuperModel})
        winningModel_config = winningModel.get_config()
        winningModel_name = winningModel_config['name']
        img_input_shape = winningModel_config['layers'][0]['config']['batch_input_shape'][1:]

        print("Loaded model: %s" %winningModel_name)

        listOfNormelizedImgs = []
        listOfImageTitles = []
        listOfImagePaths = []

        for file in listOfImageFiles:
            img, imgTitle = loadAndPrepareImgForPrediction(img_path=file, imgShape=img_input_shape)
            if img is not None:
                listOfNormelizedImgs.append(img)
                listOfImageTitles.append(imgTitle)
                listOfImagePaths.append(file)

        # Convert list of numpy to tensor array:
        tst_dataset = np.array(listOfNormelizedImgs)

        print("Predicting on group of: %s"%(str(tst_dataset.shape)))
        # prediction = winningModel.predict(tst_dataset)
        prediction_prob = winningModel.predict_proba(tst_dataset)

        # print(classDict)
        imageProbailitiesList = {}

        for i, pred in enumerate(prediction_prob):
            imgLabel = listOfImageTitles[i]
            imgPath = listOfImagePaths[i]
            imageProbailitiesList[imgLabel] = {'imgPath':imgPath}
            probDict={}
            for i, prob in enumerate(pred):
                probDict[classDict[i]] = round(prob, 3)
            # print(imgLabel)
            # print(probDict)
            imageProbailitiesList[imgLabel]['Classifications'] = probDict

        return imageProbailitiesList


if __name__ == "__main__":

    isSafeForShow = False
    showOnlyMatches = False

    basePath = os.path.join(app.config['BASE_FOLDER'], 'NudityDetector')
    classDictionaryPath = os.path.join(basePath, 'Classes', 'ClassDictionary.json')
    winningModelPath = os.path.join(basePath,r'Models\ModelOutput - NudiyDetector_Draft2\NudiyDetector_Draft2 - Accuracy 0.6.hdf5')

    listOfImageFiles = [
        r'C:\Users\omerro\Google Drive\Data Science Projects\OmerPortal\omerprojects\static\uploads\ERAN_RESCUE.jpg'
        , r'C:\Users\omerro\Google Drive\Data Science Projects\OmerPortal\omerprojects\static\uploads\Wanda_HOuse.png'
        , r'C:\Users\omerro\Google Drive\Data Science Projects\OmerPortal\omerprojects\static\uploads\147304658_10222621474379592_5661367041399207659_o.jpg'
        , r"C:\Users\omerro\Google Drive\Data Science Projects\OmerPortal\omerprojects\NudityDetector\image for test - SFW\2020-08-04 17.43.51.jpg"
        , r"C:\Users\omerro\Google Drive\Data Science Projects\OmerPortal\omerprojects\NudityDetector\image for test - SFW\2020-08-04 17.45.01.jpg"
    ]

    imageProbailitiesList = takeImagePath_ReturnPredictions(imagesPathList=listOfImageFiles, requestedModelAbsPath=winningModelPath, classDictionaryPath=classDictionaryPath)

    print(imageProbailitiesList)
