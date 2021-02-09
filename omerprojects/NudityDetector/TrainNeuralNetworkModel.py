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


def listAllImageFilesAndTheirClasses(absoluteClassPath,classDict,saveToCSV=True):

    classImgFolder = absoluteClassPath

    print("Getting all files")
    allFiles = getListOfFiles(classImgFolder)

    # Create a dataframe for containing file information and classes
    dataFrameColumns = ['ImgId', 'ImgPath_Relative','labels','ImgPath_Absolute']
    for key, val in classDict.items():
        dataFrameColumns.append(val)

    mainClassInstructionsDict = {}

    imageId=1
    for i, filePath in enumerate(allFiles):
        if filePath.lower().endswith(('.png', '.jpg', '.jpeg','.jfif')):
            relativePath = filePath.replace(app.config['BASE_FOLDER'], '')
            imageInfo = {'ImgPath_Relative': relativePath,
                         'ImgPath_Absolute': filePath}
            listOfFolders = filePath.split('\\')
            labels=""
            for folder in listOfFolders:
                if folder in classDict.values():
                    imageInfo[folder] = 1
                    labels+=folder+','
            if labels.endswith((',')):
                labels=labels[:len(labels)-1]

            imageInfo['labels']=labels
            mainClassInstructionsDict[imageId] = imageInfo
            imageId += i

    mainClassInstructionsDF = pd.DataFrame(mainClassInstructionsDict).fillna(0)
    mainClassInstructionsDF = mainClassInstructionsDF.T.filter(items=dataFrameColumns)

    if saveToCSV:
        path_ImagePathsAndClasses = os.path.join(classImgFolder, 'ImagePathsAndClasses.csv')
        mainClassInstructionsDF.to_csv(path_or_buf=path_ImagePathsAndClasses,
                                   index_label='ImgId')
        print("Saved ImagePathsAndClasses to %s"%(path_ImagePathsAndClasses))

    return  mainClassInstructionsDF


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


if __name__ == '__main__':

    classDict = {0: 'Penis', 1: 'Vagina', 2: 'Butt', 3: 'BreastWoman', 4: 'BreastMan', 5: 'BathingSuite', 6: 'Banana', 7: 'Peach'}

    classImgFolder = os.path.join(app.config['BASE_FOLDER'],"NudityDetector/Classes")
    path_ImagePathsAndClasses = os.path.join(classImgFolder, 'ImagePathsAndClasses.csv')
    baseModelFolder = os.path.join(app.config['BASE_FOLDER'],"NudityDetector/Models")

    loadImgClassInstructions = True

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

    dataset_train, dataset_val, dataset_test = splitDataSet_train_val_test(dataFrame=mainClassInstructionsDF,val_percent=20,test_percent=10)

    target_img_shape = (256, 256, 3)
    target_size = target_img_shape[0:2]

    print("Build data generator")
    #https://vijayabhaskar96.medium.com/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24

    datagen = ImageDataGenerator(rescale=1. / 255.)
    test_datagen = ImageDataGenerator(rescale=1. / 255.)
    train_generator = datagen.flow_from_dataframe(
        dataframe=dataset_train,
        directory=None,#app.config['BASE_FOLDER'],
        x_col="ImgPath_Absolute",
        class_mode="raw",
        #y_col="labels",
        #class_mode="categorical",
        y_col=list(classDict.values()),
        batch_size=32,
        seed=42,
        shuffle=True,
        #class_mode="categorical",
        #classes=list(classDict.values()),
        target_size=target_size)
    valid_generator = test_datagen.flow_from_dataframe(
        dataframe=dataset_val,
        directory=None,  # app.config['BASE_FOLDER'],
        x_col="ImgPath_Absolute",
        class_mode="raw",
        # y_col="labels",
        # class_mode="categorical",
        y_col=list(classDict.values()),
        batch_size=32,
        seed=42,
        shuffle=True,
        # class_mode="categorical",
        # classes=list(classDict.values()),
        target_size=target_size)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=dataset_test,
        directory=None,  # app.config['BASE_FOLDER'],
        x_col="ImgPath_Absolute",
        class_mode="raw",
        # y_col="labels",
        # class_mode="categorical",
        y_col=list(classDict.values()),
        batch_size=1,
        seed=42,
        shuffle=False,
        # class_mode="categorical",
        # classes=list(classDict.values()),
        target_size=target_size)



    NSFW_VGG16_LastLayer = OmerSuperModel(name='NSFW_VGG16_LastLayer', basePath=baseModelFolder, classDictionary=classDict)

    VGG16Model = K.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=target_img_shape)
    layers_list = []
    for i, layer in enumerate(VGG16Model.layers):
        if i < len(VGG16Model.layers) - 4:
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
    NSFW_VGG16_LastLayer.add_Checkpoint()

    NSFW_VGG16_LastLayer.summary()

    def generator_wrapper(generator):
        for batch_x,batch_y in generator:
            yield (batch_x,[batch_y[:,i] for i in range(5)])


    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    #Old fir generator - Depricated
    # NSFW_VGG16_LastLayer.fit_generator(generator=generator_wrapper(train_generator),
    #                     steps_per_epoch=STEP_SIZE_TRAIN,
    # validation_data = generator_wrapper(valid_generator),
    # validation_steps = STEP_SIZE_VALID,
    # epochs = 1, verbose = 2
    # )

    NSFW_VGG16_LastLayer.fit(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=50,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID)

    test_generator.reset()
    pred = NSFW_VGG16_LastLayer.predict_generator(test_generator,
                                   steps=STEP_SIZE_TEST,
                                   verbose=1)

    pred_bool = (pred > 0.5)

    predictions = pred_bool.astype(int)
    columns = list(classDict.values())
    # columns should be the same order of y_col
    results = pd.DataFrame(predictions, columns=columns)
    results["Filenames"] = test_generator.filenames
    ordered_cols = ["Filenames"] + columns
    results = results[ordered_cols]  # To get the same column order
    results.to_csv(os.path.join(baseModelFolder,"results.csv"), index=False)

    print(results)