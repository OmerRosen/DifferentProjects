from omerprojects.NudityDetector.ImageClassifierClass import OmerSuperModel
import gc
import numpy as np
from numpy import save
from numpy import asarray
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,multilabel_confusion_matrix


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


from Snippets_Various import load_img_omer,resize_image,JsonEncoder,loadClassDict,display_images_in_plot
from Snippets_ImageClass import OmerImageClass,augment_img_Class
from Snippets_SuperModelClass import OmerSuperModel
import gc
import os
import numpy as np
from numpy import save
from numpy import asarray
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,multilabel_confusion_matrix
import json
import tensorflow as tf


import tensorflow.keras as K
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
#from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model, Model,Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy,Hinge

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.metrics import CategoricalAccuracy,Accuracy,BinaryCrossentropy,CosineSimilarity

import tensorflow_addons as tfa
from tensorflow_addons.optimizers import Lookahead,RectifiedAdam,AdamW




def loadModel(winningModelPath="omerprojects/NudityDetector/Models/winningModelPath.hdf5",classDictionaryPath="omerprojects/NudityDetector/Classes/ClassDictionary.json"):
    print ("winningModelPath: "+winningModelPath)
    NSFW_Model = load_model(winningModelPath,compile=False,
                            custom_objects={'OmerSuperModel': OmerSuperModel})

    NSFW_Model.classDictionary,_ = loadClassDict(classDictionaryPath)
    print("ClassDict: "+str(NSFW_Model.classDictionary))
    return NSFW_Model

def predictionToString(predictProbability,classDict,thresholdForPositive=0.5):
    listOfPositives=[]
    imgTitle=""
    wasMatchFound=False
    for i in range(len(predictProbability)):
        className=classDict[i]
        prediction=round(predictProbability[i],2)
        if prediction>thresholdForPositive:
            print("Img %s: Positive: %s(%s) " % (i,className, prediction))
            imgTitle+="%s(%s) "%(className,prediction)
            listOfPositives.append(className)
            wasMatchFound=True
        else:
            print("Img %s: Did not pass: %s(%s) "%(i,className,prediction))
    return imgTitle,wasMatchFound

def loadAndPredictImages(Model,listOfImagePaths,img_Shape,thresholdForPositive=0.5,batchSize=10,SFWMode=False,displaySample=False):
    fileIndex=0
    totalImages=len(listOfImagePaths)
    listOfImageMatches=[]
    listOfImageTitles=[]
    listOfImageMatchesPath=[]
    while fileIndex<totalImages:
        listOfImagesNorm = []
        listOfPaths=[]
        nexBatchRunSize=min(batchSize,len(listOfImagePaths)-fileIndex)
        for batchIndex in range(nexBatchRunSize):
            imgPath=listOfImagePaths[fileIndex]
            img = load_img_omer(imgPath)
            if img is not None:
                img = resize_image(img,desired_width=img_Shape,desired_height=img_Shape)
                img = img/255
                listOfImagesNorm.append(img)
                listOfPaths.append(imgPath)
            fileIndex+=1
            #print("Index: %s, batch run %s  Path %s"%(fileIndex,batchIndex,imgPath))
        print("Finshes batch. %s images will be evaluated"%len(listOfImagesNorm))
        print("Predicting batch")
        predictProbabilities = Model.predict(np.array((listOfImagesNorm)))

        # Get all possibilities:
        for predIndex,predictProbability in enumerate(predictProbabilities):
            imgTitle,wasMatchFound = predictionToString(predictProbability, classDict=classDict, thresholdForPositive=0.6)
            if wasMatchFound:
                pathIM=listOfPaths[predIndex]
                listOfImageMatches.append(listOfImagesNorm[predIndex])
                listOfImageTitles.append(imgTitle)
                listOfImageMatchesPath.append(pathIM)
                print("Match was found: '%s' - for %s"%(imgTitle,pathIM))
        fileIndex+=1
        if displaySample:
            display_images_in_plot(listOfImageMatches[0:15], listOfImageTitles[0:15],images_per_row=3)


def getListOfFiles(dirName):
    imageTypeFiles = []
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if fullPath.find('.jpg')!=-1 or fullPath.find('.jpeg')!=-1 or fullPath.find('.png')!=-1 or fullPath.find('.jpeg')!=-1:
                imageTypeFiles.append(fullPath)
            allFiles.append(fullPath)
    print("%s image type files were found out of %s files in total"%(len(imageTypeFiles),len(allFiles)))
    return imageTypeFiles

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

"""
NSFW_Model = loadModel(winningModelPath=winningModelPath)

allImageTypeFiles = getListOfFiles(selectedPathToScan)
print(len(allImageTypeFiles))

random.shuffle(allImageTypeFiles)

loadAndPredictImages(Model=NSFW_Model,img_Shape=80, listOfImagePaths=allImageTypeFiles[0:60],thresholdForPositive=0.1,batchSize=5,SFWMode=False)
"""

print(os.getcwd())

modelPath = os.path.join(os.getcwd(),'ResnetModel_LastLayer_Shape_224_Accuracy_60.00625.hdf5')

#model = load_model(compile=False,filepath=modelPath,custom_objects={'OmerSuperModel': OmerSuperModel})
#model.summary()






if __name__=="__main__":

    listOfImageFiles = [
        r'C:\Users\omerro\Google Drive\Data Science Projects\OmerPortal\omerprojects\static\uploads\2020-12-18_13.29.56.jpg'
        ,r'C:\Users\omerro\Google Drive\Data Science Projects\OmerPortal\omerprojects\static\uploads\Charge_statement.jpg'
        ,r'C:\Users\omerro\Google Drive\Data Science Projects\OmerPortal\omerprojects\static\uploads\David_-_Makur_Anonimi.png'
    ]

    isSafeForShow=False
    showOnlyMatches=False

    winningModelFolderName = 'ResnetModel_LastBlock'
    winningModelFileName = 'ResnetModel_LastBlock_Shape_224_Accuracy_74.hdf5'

    ## All paths
    basePath = r'C:\Users\omerro\Google Drive\Data Science Projects\OmerPortal\omerprojects\NudityDetector'
    classDictionaryPath = os.path.join(basePath, 'Classes', 'ClassDictionary.json')
    baseModelPath = os.path.join(basePath, 'Models')
    selectedPathToScan = 'D:\Dropbox\Camera Uploads'  # 'D:\Google Drive\Harvard HW\Final Project\Classes\Butt'

    winningModel_loadPath = os.path.join(baseModelPath, "ModelOutput - "+winningModelFolderName,winningModelFileName)

    classDict, reverseClassDict = loadClassDict(ClassDictPath=classDictionaryPath)

    NSFW_Model = OmerSuperModel(name=winningModelFolderName, basePath=baseModelPath)

    NSFW_Model.load_model_forPredictions(modelPath=winningModel_loadPath, classDictionary=classDict,
                                         oldModelName="2020-8-7 7_44 - ResnetModel_LastBlock")

    listOfImagesNorm, listOfValidImgPaths = NSFW_Model.preProcessImages(listOfImagePaths=listOfImageFiles, img_Shape=224, batchSize=10 )

    listOfImageMatchesPath, listOfImageTitles, matchList, classesPredictItem = NSFW_Model.loadAndPredictImages(
        listOfImagePaths=listOfImageFiles,
        img_Shape=224, thresholdForPositive=0.5,
        batchSize=50, showOnlySFW=isSafeForShow,
        showOnlyMatches=showOnlyMatches,
        displaySample=False)

    # model = load_model(filepath=winningModel_loadPath, custom_objects={'OmerSuperModel': OmerSuperModel})


    print(listOfImageMatchesPath, listOfImageTitles, matchList, classesPredictItem)
    # listOfImageMatchesPath, listOfImageTitles, matchList, classesPredictItem = NSFW_Model.loadAndPredictImages(
    #     listOfImagePaths=listOfImageFiles,
    #     img_Shape=224, thresholdForPositive=0.5,
    #     batchSize=50, showOnlySFW=isSafeForShow,
    #     showOnlyMatches=showOnlyMatches,
    #     displaySample=False)