from omerprojects.NudityDetector.ImageClassifierClass import OmerSuperModel,load_dataset
import json
import pandas as pd

import tensorflow.keras as K
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model, Model,Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.losses import CategoricalCrossentropy,Hinge
from keras_lookahead import Lookahead
from tensorflow.keras.applications.vgg16 import VGG16
import progressbar
import os
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    bar = progressbar.ProgressBar(max_value=len(listOfFile),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    allFiles = list()
    # Iterate over all the entries
    for i, entry in enumerate(listOfFile):
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
        bar.update(i)
    return allFiles


if __name__ == '__main__':

    classDict = {0: 'Penis', 1: 'Vagina', 2: 'Butt', 3: 'BreastWoman', 4: 'BreastMan', 5: 'BathingSuite', 6: 'Banana', 7: 'Peach'}

    classImgFolder = r"C:\Users\omerro\Google Drive\Harvard HW\Final Project\Classes"

    print("Getting all files")
    allFiles = getListOfFiles(classImgFolder)

    #Create a dataframe for containing file information and classes
    dataFrameColumns = ['ImgId', 'ImgPath']
    for key, val in classDict.items():
        dataFrameColumns.append(val)



    mainClassInstructionsDict = {}

    for i, filePath in enumerate(allFiles):
        imageId = i
        imageInfo = {'ImgPath':filePath}
        listOfFolders = filePath.split('\\')
        for folder in listOfFolders:
            if folder in classDict.values():
                imageInfo[folder]=1

        mainClassInstructionsDict[imageId] = imageInfo

    mainClassInstructionsDF = pd.DataFrame(mainClassInstructionsDict).fillna(0)
    mainClassInstructionsDF = mainClassInstructionsDF.T.filter(items=dataFrameColumns)

    print(mainClassInstructionsDF.head(10))

    mainClassInstructionsDF.to_csv(index_label='ImgId')








    # with open(r'C:\Users\omerro\Google Drive\Harvard HW\Final Project\Classes\ClassDictionary.json', 'r') as jsonFile:
    #     classDict = json.load(jsonFile)
    #
    # # Reverse Dict
    # reverseClassDict = {}
    # for key, val in classDict.items():
    #     reverseClassDict[val] = int(key)
    #
    # # Convert ClassDict keys to ints:
    # for key, val in reverseClassDict.items():
    #     classDict[val] = classDict.pop(str(val))
    #
    # # Fill in gaps for dict:
    # for i in range(max(classDict.keys())):
    #     if classDict.get(i) is None:
    #         classDict[i] = None
    #
    # print(classDict)
    # print(reverseClassDict)
    #
    # X_train, y_train_catg, X_val, y_val_catg, X_test, y_test_catg = load_dataset(
    #     r'C:\Users\omerro\Google Drive/Harvard HW/Final Project/Classes/NSFW_ListOfImages.npy',
    #     r'C:\Users\omerro\Google Drive/Harvard HW/Final Project/Classes/NSFW_ListOfLabels.npy', classDict, sampleDataset=True)
    #
    # imageShape = X_train.shape[1:]
    #
    # print("Training images: %s" % (len(X_train)))
    # print("Validation images: %s" % (len(X_val)))
    # print("Test images: %s" % (len(X_test)))
    # print("Images shape: %s" % str(imageShape))
    #
    # baseFolder = 'C:\Users\omerro\Google Drive/Harvard HW/Final Project/Models'
    #
    # NSFW_VGG16_LastLayer = OmerSuperModel(X_train, y_train_catg, X_val, y_val_catg, X_test, y_test_catg,
    #                                       name='NSFW_VGG16_LastLayer', basePath=baseFolder, classDictionary=classDict)
    #
    # VGG16Model = K.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=imageShape)
    # layers_list = []
    # for i, layer in enumerate(VGG16Model.layers):
    #     if i < len(VGG16Model.layers) - 4:
    #         layer.trainable = False
    #     layers_list.append({
    #         'layer': type(layer).__name__,
    #         'trainable': layer.trainable
    #     })
    # layers_df = pd.DataFrame(layers_list)
    #
    # NSFW_VGG16_LastLayer.add(VGG16Model)
    # NSFW_VGG16_LastLayer.add(Flatten())
    # # Add 2 dense layer
    # NSFW_VGG16_LastLayer.add_dense_layer(isLastLayer=False, addBatchNorm=False, units=256, activation="relu",
    #                                      addRegularizer=None)
    # # NSFW_VGG16_LastLayer.add_dense_layer(isLastLayer=False,addBatchNorm=False,units=256,activation="relu",addRegularizer=None)
    # # output layer
    # NSFW_VGG16_LastLayer.add_dense_layer(isLastLayer=True, )
    # NSFW_VGG16_LastLayer.compile_model(optimizer='Adam', learning_rate=0.006, momentum=0.0)
    # NSFW_VGG16_LastLayer.add_EarlyStopping(patience=20)
    # NSFW_VGG16_LastLayer.add_Checkpoint()
    #
    # NSFW_VGG16_LastLayer.summary()