from omerprojects.__init__ import  app
from omerprojects.NudityDetector.ImageClassifierClass import OmerSuperModel,load_dataset
import json
import pandas as pd

import tensorflow.keras as K
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

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


    classDict = {0: 'Penis', 1: 'Vagina', 2: 'Butt', 3: 'BreastWoman', 4: 'BreastMan', 5: 'BathingSuite', 6: 'Banana', 7: 'Peach'}
    target_img_shape=(256, 256, 3)
    batch_size = 64

    classImgFolder = os.path.join(app.config['BASE_FOLDER'],"NudityDetector/Classes")
    path_ImagePathsAndClasses = os.path.join(classImgFolder, 'ImagePathsAndClasses.csv')
    baseModelFolder = os.path.join(app.config['BASE_FOLDER'],"NudityDetector/Models")

    loadImgClassInstructions = False

    # Reverse Dict
    reverseClassDict = {}
    for key, val in classDict.items():
        reverseClassDict[val] = int(key)

    # # Convert ClassDict keys to ints:
    # for key, val in reverseClassDict.items():
    #     classDict[val] = classDict.pop(str(val))
    #
    # # Fill in gaps for dict:
    # for i in range(max(classDict.keys())):
    #     if classDict.get(i) is None:
    #         classDict[i] = None

    if loadImgClassInstructions is False or not os.path.exists(path_ImagePathsAndClasses):
        mainClassInstructionsDF = listAllImageFilesAndTheirClasses(absoluteClassPath=classImgFolder,
                                                                   classDict=classDict,
                                                                   saveToCSV=True)
    else:
        mainClassInstructionsDF = pd.read_csv(path_ImagePathsAndClasses)
        print("Loaded ImagePathsAndClasses from %s" % (path_ImagePathsAndClasses))

    NSFW_VGG16_LastLayer = OmerSuperModel(name='NSFW_VGG16_LastLayer_DataGenerator', basePath=baseModelFolder,
                                          classDictionary=classDict)


    VGG16Model = K.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=target_img_shape)
    layers_list = []
    for i, layer in enumerate(VGG16Model.layers):
        if i < len(VGG16Model.layers) - 1:
            layer.trainable = False
        layers_list.append({
            'layer': type(layer).__name__,
            'trainable': layer.trainable
        })
    layers_df = pd.DataFrame(layers_list)

    NSFW_VGG16_LastLayer.add(VGG16Model)
    NSFW_VGG16_LastLayer.add(Flatten())
    # Add 2 dense layer
    NSFW_VGG16_LastLayer.add_dense_layer(isLastLayer=False, addBatchNorm=False, units=256, activation="relu",
                                         addRegularizer=None)
    # NSFW_VGG16_LastLayer.add_dense_layer(isLastLayer=False,addBatchNorm=False,units=256,activation="relu",addRegularizer=None)
    # output layer
    NSFW_VGG16_LastLayer.add_dense_layer(isLastLayer=True, )
    NSFW_VGG16_LastLayer.compile_model(optimizer='Adam', learning_rate=0.006, momentum=0.0)
    NSFW_VGG16_LastLayer.add_EarlyStopping(patience=7)
    #NSFW_VGG16_LastLayer.add_Checkpoint()

    NSFW_VGG16_LastLayer.summary()


    test_mainClassInstructionsDF=mainClassInstructionsDF[:500]
    NSFW_VGG16_LastLayer.buildDataGeneretor(mainClassInstructionsDF=test_mainClassInstructionsDF,
                                            target_img_shape=target_img_shape,
                                            batch_size=batch_size)

    NSFW_VGG16_LastLayer.train_or_load(trainRegardless=True,
                                       epochs=2,
                                       batch_size=batch_size)

    print(os.path.join(NSFW_VGG16_LastLayer.basePath, "Matrics_Output_Compare.json"))
    NSFW_VGG16_LastLayer.compare_all_models()
    NSFW_VGG16_LastLayer.compare_json.sort_values(by=["accuracy"], ascending=False).head(10)

    NSFW_VGG16_LastLayer.test_real_images(
        realImagePath='/content/drive/My Drive/Harvard HW/Final Project/ImagesForTest', threshold=0.5, batchSize=10,
        modelImgShape=NSFW_VGG16_LastLayer.image_shape[0], showOnlyMatches=False, showSFW=False)