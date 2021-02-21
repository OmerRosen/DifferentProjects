import numpy as np
from numpy import save
from numpy import asarray
from Snippets_Various import load_img_omer, resize_image, display_images_in_plot, splitDataSet_train_val_test,JsonEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, multilabel_confusion_matrix
# import tensorflow as tf


import tensorflow.keras as K
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, \
    Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy, Hinge
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.metrics import CategoricalAccuracy, Accuracy, BinaryCrossentropy, CosineSimilarity

import tensorflow_addons as tfa
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam, AdamW

from sklearn import model_selection

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

""" Snippet to quickly display a list of images in a plot """

"""
Version 1.0.2
Edits:
Added Load Specific model
For Compile - Add measure_metrics parameter: ['accuracy']/[categorical_accuracy]/[iou, iou_thresholded]
"""


class OmerSuperModel(Sequential):
    def __init__(self, X_train=[], y_train=[], X_val=[], y_val=[], X_test=[], y_test=[], classDictionary=[],
                 name="Model", basePath=""):
        super().__init__(name=name)
        self.basePath = basePath
        self.modelName = name
        self.hdf5_path = None
        self.classDictionary = classDictionary
        self.numOfClasses = len(classDictionary)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.classDictionary = classDictionary
        self.training_results = None
        self.execution_time = None
        self.save_path = os.path.join(self.basePath, "ModelOutput - %s" % self.modelName)
        self.save_plot_path = None
        self.fit_callback_list = []
        self.compare_json = None
        self.modelmetrics = {}
        self.batch_size = 34
        self.measurment_metric_name = 'accuracy'
        self.modelTempName = name
        print("New Sequential type model was created: %s" % (self.modelName))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if X_train != []:
            self.image_shape = X_train.shape[1:]
        else:
            print('Model Was Created without train values')
        self.classification_report = []
        self.evaluation_results = None
        self.isSegmentedModel = False

        # For Data generator:
        self.DataGenerator = 0

        self.test_generator = None
        self.valid_generator = None
        self.train_generator = None
        self.STEP_SIZE_TRAIN = None
        self.STEP_SIZE_VALID = None
        self.STEP_SIZE_TEST = None

    def normalize_data(self, alsoFlatten=False):
        self.X_train = self.X_train.astype("float") / 255.0
        self.X_val = self.X_val.astype("float") / 255.0
        self.X_test = self.X_test.astype("float") / 255.0
        print("Data normalized successfully")
        if alsoFlatten:
            imgCount, img_H, img_W, channels = self.X_train.shape
            new_shape = img_H * img_W * channels
            self.X_train = self.X_train.reshape(new_shape)
            self.X_val = self.X_val.reshape(new_shape)
            self.X_test = self.X_test.reshape(new_shape)
            self.image_shape = new_shape

    def display_input_data(self):
        print("Inputs x:")
        print("X_train shape:", self.X_train.shape)
        print("X_val shape:", self.X_val.shape)
        print("X_test shape:", self.X_test.shape)
        print("Outputs y:")
        print("y_train shape:", self.y_train.shape)
        print("y_val shape:", self.y_val.shape)
        print("y_test shape:", self.y_test.shape)

        print("Inputs:\n", self.X_train[0][:5, :5, 0])
        print("Outputs:\n", self.y_train[:5])

    def add_input_layer(self):
        super().add(Input(shape=self.image_shape))
        print("Input layer was added. Shape: %s" % (str(self.image_shape)))

    def add_dense_layer(self, isLastLayer=False, addBatchNorm=False, units=128, activation="relu", addRegularizer=None):
        regulizer = None
        if addRegularizer:
            regulizer = l2(addRegularizer)
            print('Added regulizer to layer: %s' % (addRegularizer))
        if isLastLayer:  # If last layer, unit should reflect the number of classes
            super().add(Dense(units=self.numOfClasses, activation='softmax'))
            print("Last output layer was added. units: %s, Activation: %s" % (self.numOfClasses, 'sigmoid'))
        else:
            # activation='softmax'
            super().add(Dense(units=units, kernel_regularizer=regulizer))  # , activation=activation
            if addBatchNorm:
                super().add(BatchNormalization())
            super().add(Activation(activation))

            print("Dense layer was added. units: %s, Activation: %s" % (units, activation))

    # FFFF
    def add_convolution_layer(self, addPooling=True,
                              flattenLayer=False, addBatchNorm=False,
                              isFirst=False, filters=128, kernelSize=(3, 3), pool_size=(2, 2)
                              , conv_strides=(1, 1), pool_strides=(2, 2), activation="relu"
                              , addRegularizer=None):
        regulizer = None
        if addRegularizer:
            regulizer = l2(addRegularizer)
            print('Added regulizer to layer: %s' % (addRegularizer))
        if isFirst:
            super().add(Conv2D(filters=filters, input_shape=self.image_shape,
                               kernel_size=kernelSize, strides=conv_strides, padding='same',
                               kernel_regularizer=regulizer))
            if addBatchNorm:
                super().add(BatchNormalization())
            super().add(Activation(activation))
        else:
            super().add(Conv2D(filters=filters, kernel_size=kernelSize, strides=conv_strides, padding='same',
                               kernel_regularizer=regulizer))
            if addBatchNorm:
                super().add(BatchNormalization())
            super().add(Activation(activation))
        print("Convolution layer was added. filters: %s, kernelSize: %s, kernelStride: %s, activation: %s" % (
            filters, kernelSize[1], conv_strides[0], activation))
        if addPooling:
            super().add(MaxPooling2D(pool_size=pool_size, strides=pool_strides))
            print("Pooling layer was added. pool_size: %s, pool_strides: %s" % (pool_size[0], pool_strides[0]))
        if flattenLayer:
            super().add(Flatten())
            print("Convolution layer was flattened")

    def add_EarlyStopping(self, patience=10, monitor='val_loss', min_delta=0):
        earlyStopping = EarlyStopping(patience=patience, monitor=monitor, min_delta=min_delta)
        self.fit_callback_list.append(earlyStopping)
        print(
            "EarlyStopping was added to model: patience:%s, ,monitor=%s ,min_delta:%s" % (patience, monitor, min_delta))

    def add_Checkpoint(self, patience=10, monitor='val_loss', min_delta=0):
        # DDDD
        checkpointFolderPath = os.path.join(self.save_path, 'Checkpoint')
        if not os.path.exists(checkpointFolderPath):
            os.makedirs(checkpointFolderPath)
        checkpointFilePath = os.path.join(checkpointFolderPath, '%s - Epoc {epoch:02d} - Val_%s {val_%s:.2f}.hdf5' % (
            # checkpointFilePath = os.path.join(checkpointFolderPath, '%s - Epoc {epoch:02d} - Val_%s {Val_%s:.2f}.hdf5' % (
            self.modelName, self.measurment_metric_name, self.measurment_metric_name))
        # checkpointFilePath = os.path.join(checkpointFolderPath,self.save_path,self.modelName+".hdf5")
        checkpoint = ModelCheckpoint(filepath=checkpointFilePath, monitor='val_%s' % self.measurment_metric_name,
                                     save_best_only=True, mode='max')
        self.fit_callback_list.append(checkpoint)
        print("Checkpoint was added to model. File: %s" % (checkpointFilePath))

    def add_Dropout_layer(self, rate=0.5):
        super().add(Dropout(rate=rate))
        print("Dropout layer was added. Rate: %s" % (rate))

    def compile_model(self, optimizer='SGD', learning_rate=0.01, momentum=0.00, loss='binary_crossentropy',
                      warmup_proportion=0.1, total_steps=10000, min_lr=1e-5, measurment_metric_name='accuracy'):
        # MMMM
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.warmup_proportion = warmup_proportion
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.measurment_metric_name = measurment_metric_name

        if type(optimizer) == str:
            # Assign Optimizers based on selection
            optimizerType = optimizer.upper()
            if optimizerType == 'SGD':
                optimizerClass = SGD(learning_rate=learning_rate, momentum=momentum)
            elif optimizerType == 'ADAM':
                optimizerClass = Adam(learning_rate=learning_rate)
            elif optimizerType == 'RMSPROP':
                optimizerClass = RMSprop(learning_rate=learning_rate)
            elif optimizerType == 'RADAM':
                optimizerClass = RectifiedAdam(learning_rate=learning_rate, warmup_proportion=warmup_proportion,
                                               decay=min_lr)  # total_steps=total_steps,min_lr=min_lr)
            elif optimizerType == 'LOOKAHEAD':
                optimizerClass = RectifiedAdam(learning_rate=learning_rate, warmup_proportion=warmup_proportion,
                                               decay=min_lr)  # total_steps=total_steps,min_lr=min_lr)
                optimizerClass = Lookahead(optimizerClass)
            else:
                print("Optimizer is NOT in string list.")
        else:
            optimizerClass = optimizer
            optimizerType = "ManualFunc"
        # Assign measure matrics based on selection:
        measurment_metric = ''
        if measurment_metric_name.lower() == 'accuracy':
            measurment_metric = Accuracy
            self.measurment_metric_name = 'accuracy'
        elif measurment_metric_name.lower() == 'categorical_accuracy':
            measurment_metric = [CategoricalAccuracy]
            self.measurment_metric_name = 'categorical_accuracy'
        # elif measurment_metric_name.lower() == 'iou':
        #     measurment_metric = [iou]
        #     self.measurment_metric_name = 'iou'
        # elif measurment_metric_name.lower() == 'iou_thresholded':
        #     measurment_metric = [iou_thresholded]
        #     self.measurment_metric_name = 'iou_thresholded'

        super().compile(optimizer=optimizerClass, loss=loss, metrics=self.measurment_metric_name)
        # super().compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
        self.optimizerType = optimizerType
        print("Model was complied. optimizer: %s, learning_rate: %s, momentum: %s" % (
            self.optimizerType, self.learning_rate, self.momentum))

    """def compile_model_segment(self,optimizer='SGD',learning_rate=0.01,momentum=0.00,loss='categorical_crossentropy',warmup_proportion=0.1,total_steps=10000,min_lr=1e-5):
      optimizerType=optimizer.upper()
      if optimizerType=='SGD':
        optimizerClass = SGD(learning_rate=learning_rate,momentum=momentum)
      elif optimizerType=='ADAM':
        optimizerClass = Adam(learning_rate=learning_rate)
      elif optimizerType=='RMSPROP':
        optimizerClass = RMSprop(learning_rate=learning_rate)
      elif optimizerType=='RADAM':
        optimizerClass = RAdam(learning_rate=learning_rate,warmup_proportion=warmup_proportion,total_steps=total_steps,min_lr=min_lr)
      elif optimizerType=='LOOKAHEAD':
        optimizerClass = RAdam(learning_rate=learning_rate,warmup_proportion=warmup_proportion,total_steps=total_steps,min_lr=min_lr)
        optimizerClass = Lookahead(optimizerClass)
      super().compile(optimizer=optimizerClass, loss=loss, metrics=[iou, iou_thresholded])
      #super().compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
      self.learning_rate = learning_rate
      self.momentum = momentum
      self.optimizerType = optimizerType

      self.isSegmentedModel=True
      print("Model was complied as segmented model. optimizer: %s, learning_rate: %s, momentum: %s"%(self.optimizerType,self.learning_rate,self.momentum))
    """

    def buildDataGeneretor(self, mainClassInstructionsDF, target_img_shape=(256, 256, 3), batch_size=64,
                           pathColName='ImgPath_Absolute'):

        # Receivce a df file containing all mainClassInstructionsDF
        # Location col name: ImgPath_Absolute

        testPath = mainClassInstructionsDF[pathColName].values[0]
        if not os.path.exists(testPath):
            print("Could not locate sample image from mainClassInstructionsDF")
            print("Img path: %s" % (testPath))

        start_time = time.time()

        dataset_train, dataset_val, dataset_test = splitDataSet_train_val_test(dataFrame=mainClassInstructionsDF,
                                                                               val_percent=20, test_percent=10)
        target_size = target_img_shape[0:2]

        # self.y_test = dataset_test.filter(items=self.classDictionary.values()).values
        self.X_test = dataset_test.filter(items=[pathColName]).values

        print("Build data generator")
        # https://vijayabhaskar96.medium.com/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24

        datagen = ImageDataGenerator(rescale=1. / 255.)
        test_datagen = ImageDataGenerator(rescale=1. / 255.)
        train_generator = datagen.flow_from_dataframe(
            dataframe=dataset_train,
            directory=None,
            x_col=pathColName,
            class_mode="raw",
            y_col=list(self.classDictionary.values()),
            batch_size=batch_size,
            seed=42,
            shuffle=True,
            target_size=target_size)
        valid_generator = test_datagen.flow_from_dataframe(
            dataframe=dataset_val,
            directory=None,  # app.config['BASE_FOLDER'],
            x_col=pathColName,
            class_mode="raw",
            y_col=list(self.classDictionary.values()),
            batch_size=batch_size,
            seed=42,
            shuffle=True,
            target_size=target_size)
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=dataset_test,
            directory=None,  # app.config['BASE_FOLDER'],
            x_col=pathColName,
            class_mode="raw",
            y_col=list(self.classDictionary.values()),
            batch_size=1,
            seed=42,
            shuffle=False,
            target_size=target_size)

        self.test_generator = test_generator
        self.valid_generator = valid_generator
        self.train_generator = train_generator

        self.y_test = self.test_generator.labels

        def generator_wrapper(generator):
            for batch_x, batch_y in generator:
                yield (batch_x, [batch_y[:, i] for i in range(5)])

        self.STEP_SIZE_TRAIN = self.test_generator.n // self.test_generator.batch_size
        self.STEP_SIZE_VALID = self.valid_generator.n // self.valid_generator.batch_size
        self.STEP_SIZE_TEST = self.train_generator.n  # // self.train_generator.batch_size

        self.DataGenerator = 1

        print("test_generator: %s records. Shape: %s" % (self.test_generator.n, str(self.y_test.shape)))
        print("valid_generator: %s records" % (self.valid_generator.n))
        print("train_generator: %s records" % (self.train_generator.n))

    def trainModel(self, batch_size=34, epochs=30, saveToFile=True, alsoTestModel=True, plotResults=True,
                   saveTempModel=True, target_img_shape=(256, 256, 3)):

        self.image_shape = target_img_shape

        start_time = time.time()
        callBacks = None
        if self.fit_callback_list != []:
            callBacks = self.fit_callback_list

        if self.DataGenerator == 1:

            training_results = super().fit(
                self.train_generator,
                steps_per_epoch=self.STEP_SIZE_TRAIN,
                epochs=epochs,
                validation_data=self.valid_generator,
                validation_steps=self.STEP_SIZE_VALID,
                callbacks=callBacks

            )

        else:  # Regular dataset training
            training_results = super().fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_val, self.y_val),
                batch_size=batch_size,
                # Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32
                epochs=epochs,
                # Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
                callbacks=callBacks
                # ,verbose=1
            )
            self.execution_time = (time.time() - start_time) / 60.0
            print("Training execution time (mins)", self.execution_time)

        self.training_results = training_results

        self.batch_size = batch_size
        self.epochs = epochs

        model_train_history = self.training_results.history
        self.model_train_history = model_train_history
        # Get the number of epochs the training was run for
        num_epochs = len(model_train_history["loss"])

        now = datetime.datetime.now()
        self.save_plot_path = os.path.join(self.save_path, "%s-%s-%s %s_%s - %s - train_plot.png" % (
            now.year, now.month, now.day, now.hour, now.minute, self.modelName))

        # Save model metric so that it will not be overriden:
        if saveTempModel:
            self.modelTempName = self.modelName + " - %s-%s-%s %s_%s" % (
                now.year, now.month, now.day, now.hour, now.minute)
        else:
            self.modelTempName = self.modelName

        if self.isSegmentedModel:
            # Plot training results
            fig = plt.figure(figsize=(20, 10))
            axs = fig.add_subplot(1, 2, 1)
            axs.set_title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # Plot all metrics
            for metric in ["loss", "val_loss"]:
                axs.plot(np.arange(0, num_epochs), model_train_history[metric], label=metric)
            axs.legend()

            axs = fig.add_subplot(1, 2, 2)
            axs.set_title('IOU')
            plt.xlabel('Epoch')
            plt.ylabel('IOU')
            # Plot all metrics
            for metric in ["iou", "val_iou"]:
                axs.plot(np.arange(0, num_epochs), model_train_history[metric], label=metric)
            axs.legend()
            if saveToFile:
                plt.savefig(self.save_plot_path)
            # plt.show()
        else:
            # Plot training results
            fig = plt.figure(figsize=(20, 10))
            axs = fig.add_subplot(1, 2, 1)
            axs.set_title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # Plot all metrics
            for metric in ["loss", "val_loss"]:
                axs.plot(np.arange(0, num_epochs), model_train_history[metric], label=metric)
            axs.legend()

            axs = fig.add_subplot(1, 2, 2)
            axs.set_title(self.measurment_metric_name)
            plt.xlabel('Epoch')
            plt.ylabel(self.measurment_metric_name)
            # Plot all metrics
            for metric in [self.measurment_metric_name, "val_" + self.measurment_metric_name]:
                axs.plot(np.arange(0, num_epochs), model_train_history[metric], label=metric)
            axs.legend()
            if saveToFile:
                plt.savefig(self.save_plot_path)
            # plt.show()

        if alsoTestModel:
            self.testModel(showSampleTestSize=None, printResults=False)

        return training_results

    def testModel(self, showSampleTestSize=None, printResults=True):
        if self.DataGenerator == 1:
            # print("Test model: STEP_SIZE_TEST: %s" % (self.STEP_SIZE_TEST))
            test_predictions = super().predict(self.test_generator,
                                               steps=self.STEP_SIZE_TEST,  # self.STEP_SIZE_TEST,
                                               max_queue_size=100000,
                                               verbose=1)
            # print("test_predictions: %s" % (test_predictions.shape))
            # print("self.y_test: %s" % (self.y_test.shape))
        else:
            test_predictions = super().predict(self.X_test)
        test_predictions_flattened = np.argmax(test_predictions, axis=1)
        y_test_flattened = np.argmax(self.y_test, axis=1)
        listOfImagesAndValues = []
        for i, prediction in enumerate(test_predictions):
            if self.DataGenerator == 1:  # If image generator - Load image from instruction path
                image = load_img_omer(self.X_test[i][0])
            else:
                image = self.X_test[i][0]  # If direct image - Image from X_test
            maxScore = np.max(test_predictions[i])
            maxIndex = np.argmax(test_predictions[i])
            indexValue = self.classDictionary[maxIndex]
            rowDesc = {'Image': image, 'maxScore': maxScore, 'maxIndex': maxIndex, 'label': indexValue}
            listOfImagesAndValues.append(rowDesc)
        # print("y_test_flattened: %s, test_predictions_flattened: %s" % (len(y_test_flattened), len(test_predictions_flattened)))
        cm = confusion_matrix(y_test_flattened, test_predictions_flattened)
        acc = accuracy_score(y_test_flattened, test_predictions_flattened)
        self.accuracyScore = acc

        if printResults:
            print("Correct overall: %s" % acc)
            print("Confusion matrix:\n")
            print(cm)
        if showSampleTestSize:
            listOfImages = []
            listOfLabels = []
            for i in range(showSampleTestSize):
                randInt = random.randint(0, len(test_predictions))
                img = listOfImagesAndValues[randInt]['Image']
                listOfImages.append(img)
                title = 'Predicted: %s. Score: %s' % (
                    listOfImagesAndValues[randInt]['label'], round(listOfImagesAndValues[randInt]['maxScore'], 2))
                listOfLabels.append(title)
            display_images_in_plot(listOfImages, listOfLabels)
        return listOfImagesAndValues

    def load_model(self, X_train, y_train, X_val, y_val, X_test, y_test, classDictionary):
        loadPath = os.path.join(self.save_path, self.modelName + ".hdf5")
        if os.path.exists(loadPath):
            print("Loading existing model %s from %s" % (self.modelName, loadPath))
            model = load_model(loadPath, custom_objects={'OmerSuperModel': OmerSuperModel})
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test
            self.classDictionary = classDictionary
            self.save_path = os.path.join(self.basePath, "ModelOutput - %s" % self.modelName)
            self.numOfClasses = len(classDictionary)

            # Load all information from matric:
            matric_save_path = os.path.join(self.basePath, "Matrics_Output_Compare.json")
            with open(matric_save_path) as f:
                allMatrics = json.load(f)
            modelMatric = allMatrics[self.modelName]
            print(modelMatric)
            self.batch_size = modelMatric['batch_size']
            self.epoch = modelMatric['epochs']
            self.learning_rate = modelMatric['learning_rate']
            self.optimizerType = modelMatric['optimizer']
            return model
        else:
            print("No file to upload. Please check path or train your model from start. Path: %s" % (loadPath))

    def load_model_forPredictions(self, modelPath, classDictionary, oldModelName):  # By default load existing model.
        loadPath = modelPath
        if os.path.exists(loadPath):
            print("Loading pre-trained model %s from %s" % (oldModelName, loadPath))
            model = load_model(loadPath, custom_objects={'OmerSuperModel': OmerSuperModel})
            self.X_train = []
            self.y_train = []
            self.X_val = []
            self.y_val = []
            self.X_test = []
            self.y_test = []
            self.classDictionary = classDictionary
            self.save_path = os.path.join(self.basePath, "ModelOutput - %s" % self.modelName)
            self.numOfClasses = len(classDictionary)

            # Load all information from matric:
            matric_save_path = os.path.join(self.basePath, "Matrics_Output_Compare.json")
            with open(matric_save_path) as f:
                allMatrics = json.load(f)
            modelMatric = allMatrics[oldModelName]
            print(modelMatric)
            self.batch_size = modelMatric['batch_size']
            self.epoch = modelMatric['epochs']
            self.learning_rate = modelMatric['learning_rate']
            self.optimizerType = modelMatric['optimizer']
            return model
        else:
            print("No file to upload. Please check path or train your model from start. Path: %s" % (loadPath))

    def save_model(self, saveMode=1):

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # Save the enitire model (structure + weights)
        if saveMode == 1:
            hdf5_path = os.path.join(self.save_path, "%s - Accuracy %s .hdf5"%(self.modelName,self.accuracyScore))
            print("Saving model to %s" % (self.save_path))
            super().save(hdf5_path)
        elif saveMode == 2:
            # Save only the weights
            super().save_weights(os.path.join(self.save_path, self.modelName + ".h5"))
        elif saveMode == 3:
            # Save the structure only
            model_json = super().to_json()
            with open(os.path.join(self.save_path, self.modelName + ".json"), "w") as json_file:
                json_file.write(model_json)

        model_size = os.stat(os.path.join(self.save_path, self.modelName + ".hdf5")).st_size

        # Create folder for saving model's results
        matric_save_path = self.basePath
        now = datetime.datetime.now()

        save_metrics_path = os.path.join(matric_save_path, "ModelOutput - " + self.modelName,
                                         "%s-%s-%s %s_%s - %s - train_history.json" % (
                                             now.year, now.month, now.day, now.hour, now.minute, self.modelName))
        # Save model history
        if saveMode == 1 and self.training_results.history:
            with open(save_metrics_path, "w") as json_file:
                json_file.write(json.dumps(self.training_results.history, cls=JsonEncoder))

        trainable_parameters = super().count_params()

        self.evaluate_results()

        if type(self.learning_rate != float):  # If learning rate is a class instead of a float:
            self.learning_rate = 'N/A'

        # Save model metrics
        modelmetrics = {
            "optimizer": self.optimizerType,
            "loss": self.evaluation_results[0],
            "measurment_metric_name": self.measurment_metric_name,
            "trainable_parameters": trainable_parameters,
            "execution_time": self.execution_time,
            "accuracy": self.evaluation_results[1],
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "momentum": self.momentum,
            "warmup_proportion": self.warmup_proportion,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "model_size": model_size,
        }

        self.modelmetrics = modelmetrics

        # model_metrics[self.modelName] = modelmetrics
        print(modelmetrics)
        if saveMode == 1:
            self.save_model_metrics(modelmetrics)

    def get_model_metrics(self):
        with open("model_metrics_section4a.json") as json_file:
            model_metrics = json.load(json_file)

        return model_metrics

    def save_model_metrics(self, modelmetrics=None):
        if modelmetrics is None:
            modelmetrics = self.modelmetrics
        matric_save_path = os.path.join(self.basePath, "Matrics_Output_Compare.json")
        if os.path.exists(matric_save_path):
            with open(matric_save_path) as json_file:
                model_metrics = json.load(json_file)
        else:
            model_metrics = {}

        model_metrics[self.modelTempName] = modelmetrics

        # print(matric_save_path)
        # print(modelmetrics)
        # Save the json
        with open(matric_save_path, 'w') as json_file:
            json_file.write(json.dumps(model_metrics, cls=JsonEncoder))
        print("Successfully saved model matrics to path: %s" % matric_save_path)

    def compare_all_models(self, listOfMatricesToCompare=[]):
        # Compare model metrics
        # FFFF
        matric_save_path = os.path.join(self.basePath, "Matrics_Output_Compare.json")
        allMatrics = pd.read_json(matric_save_path)

        if listOfMatricesToCompare == []:
            compare_json = allMatrics.T
            # print(compare_json)
        else:
            compare_json = allMatrics.filter(items=listOfMatricesToCompare).T

        print(
            'Matrics were compared. Please use the following command: "%s.compare_json.sort_values(by=["accuracy"],ascending=False).head(10)"' % self.modelName)
        # compare_json.sort_values(by=['accuracy'],ascending=False).head(10)
        self.compare_json = compare_json
        return self.compare_json

    def showTrainingHistory(self, saveToFile=True):
        now = datetime.datetime.now()
        self.save_plot_path = os.path.join(self.save_path, "%s-%s-%s %s_%s - %s - train_plot.png" % (
            now.year, now.month, now.day, now.hour, now.minute, self.modelName))
        plot_img_path = self.save_plot_path
        # print(plot_img_path)
        if not os.path.exists(self.save_plot_path):
            listOfFiles = os.listdir(path=self.save_path)
            listOfFiles.sort(reverse=True)
            # print(listOfFiles)
            for file in listOfFiles:
                print(file)
                if file.find('train_plot.png') != -1:
                    plot_img_path = os.path.join(self.save_path, file)
                    print("No training found, will display last saved train file: %s" % plot_img_path)
                    break
        img = cv2.imread(plot_img_path)
        fig = plt.figure(figsize=(25, 20))
        plt.title('Loaded Plot: %s' % plot_img_path)
        plt.imshow(img)

    def evaluate_results(self):

        if self.DataGenerator == 1:
            evaluation_results = super().evaluate(self.test_generator)

        else:
            evaluation_results = super().evaluate(self.X_test, self.y_test)

        self.evaluation_results = evaluation_results

    def evaluate_save_model(self, saveToFile=True):

        self.showTrainingHistory()

        # Evaluate on test data
        if self.DataGenerator == 1:
            evaluation_results = super().evaluate(self.test_generator)

        else:
            evaluation_results = super().evaluate(self.X_test, self.y_test)
        # print(evaluation_results)

        # Evaluate on test data
        test_predictions = super().predict(self.X_test, batch_size=self.batch_size)
        # print(test_predictions.shape)
        # print(test_predictions[0])

        # Classification Report
        print("classification_report")
        # print(classification_report(self.y_test.argmax(axis=1),test_predictions.argmax(axis=1), target_names=self.classDictionary.values()))
        # Save model
        classification_report_output = classification_report(self.y_test.argmax(axis=1),
                                                             test_predictions.argmax(axis=1),
                                                             target_names=self.classDictionary.values(),
                                                             output_dict=True)
        classification_report_panda = pd.DataFrame(classification_report_output)
        self.classification_report = classification_report_panda

    def train_or_load(self, trainRegardless=False, batch_size=128, epochs=50, saveToFile=True, alsoTestModel=True,
                      plotResults=True):
        modelPath = os.path.join(self.save_path, self.modelName + ".hdf5")
        if not os.path.exists(modelPath) or trainRegardless:
            self.trainModel(batch_size=batch_size, epochs=epochs, saveToFile=saveToFile, alsoTestModel=alsoTestModel,
                            plotResults=plotResults)
            self.save_model()
        else:
            self.load_model(self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test,
                            self.classDictionary)

    def loadAndPredictImages(self, listOfImagePaths, img_Shape, thresholdForPositive=0.5, batchSize=10,
                             showOnlySFW=False,
                             showOnlyMatches=False, displaySample=False, printResults=False):
        fileIndex = 0
        totalImages = len(listOfImagePaths)
        listOfImageMatches = []
        listOfImageTitles = []
        listOfImageMatchesPath = []
        classesMatchList = []
        classesPredictItem = []
        while fileIndex < totalImages:
            listOfImagesNorm = []
            listOfPaths = []
            nexBatchRunSize = min(batchSize, len(listOfImagePaths) - fileIndex)
            for batchIndex in range(nexBatchRunSize):
                imgPath = listOfImagePaths[fileIndex]
                img = load_img_omer(imgPath)
                if img is not None:
                    img = resize_image(img, desired_width=img_Shape, desired_height=img_Shape)
                    img = img / 255
                    listOfImagesNorm.append(img)
                    listOfPaths.append(imgPath)
                fileIndex += 1
                # print("Index: %s, batch run %s  Path %s"%(fileIndex,batchIndex,imgPath))
            print("Finshes batch. %s images will be evaluated" % len(listOfImagesNorm))
            print("Predicting batch")
            predictProbabilities = super().predict(np.array((listOfImagesNorm)))

            # Get all possibilities:
            for predIndex, predictProbability in enumerate(predictProbabilities):
                predictItem = {}
                if predIndex == 0:
                    if printResults: print(predictProbability)
                    if printResults: print(thresholdForPositive)
                imgTitle, wasMatchFound, isSFW, matchList, predictionStringList = self.predictionToString(
                    predictProbability, thresholdForPositive)
                if (wasMatchFound or showOnlyMatches == False) and not (isSFW == False and showOnlySFW == True):
                    pathIM = listOfPaths[predIndex]
                    listOfImageMatches.append(listOfImagesNorm[predIndex])
                    listOfImageTitles.append(imgTitle)
                    listOfImageMatchesPath.append(pathIM)
                    if wasMatchFound:
                        if printResults: print("Match was found: '%s' - for %s" % (imgTitle, pathIM))
                    predictItem['imagePath'] = pathIM
                    predictItem['imageTitle'] = imgTitle
                    predictItem['wasMatchFound'] = wasMatchFound
                    predictItem['isSFW'] = isSFW
                    predictItem['matchList'] = matchList
                    predictItem['predictionStringList'] = predictionStringList
                    classesPredictItem.append(predictItem)
                else:
                    if printResults: print("Not appended: %s - %s" % (imgTitle, pathIM))
            fileIndex += 1
        if displaySample:
            display_images_in_plot(listOfImageMatches[0:15], listOfImageTitles[0:15], images_per_row=4)
        return listOfImageMatchesPath, listOfImageTitles, classesMatchList, classesPredictItem

    def predictionToString(self, predictProbability, thresholdForPositiveInt=0.5, printResults=False):
        listOfPositives = []
        imgTitle = ""
        wasMatchFound = False
        isSFW = True
        matchList = {}
        predictionStringList = []
        for i in range(len(predictProbability)):
            try:
                className = self.classDictionary[i]
                prediction = round(predictProbability[i] * 100)
                if prediction > thresholdForPositiveInt * 100:
                    # print("Img %s: Positive: %s(%s) " % (i,className, prediction))
                    imgTitle += "%s(%s Percent) " % (className, prediction)
                    listOfPositives.append(className)
                    wasMatchFound = True
                    if className in ('Penis', 'Vagina', 'Butt', 'BreastWoman'):
                        isSFW = False
                    matchList[className] = prediction
            except:
                if printResults: print(
                    "Issue getting ClassName for value: %s - Class Dict: %s" % (i, str(self.classDictionary)))
        predictionString = str(np.round(predictProbability * 100))
        predictionStringList.append(predictionString)
        if wasMatchFound == False:
            imgTitle = "No Match Found: \n%s" % (predictionString)
        else:
            imgTitle = imgTitle + '\n%s' % (predictionString)
        return imgTitle, wasMatchFound, isSFW, matchList, predictionStringList

    def test_real_images(self, realImagePath, modelImgShape=224, threshold=0.5, batchSize=10, showOnlyMatches=False,
                         showOnlySFW=False):
        imageTypeFiles = []

        def getListOfFiles(dirName):
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
                    allFiles.append(fullPath)

            return allFiles

        listOfImagePaths = getListOfFiles(realImagePath)
        self.loadAndPredictImages(listOfImagePaths, img_Shape=modelImgShape, thresholdForPositive=threshold,
                                  batchSize=batchSize, showOnlySFW=showOnlySFW, showOnlyMatches=showOnlyMatches)

    def classificationReport_MultiLabel(self):
        y_predict = super().predict(self.X_test)
        y_predict_binary = (y_predict > 0.5)
        y_test_binary = (self.y_test > 0.5)

        testScorePerClass = dict()
        for classNum, className in self.classDictionary.items():
            testScorePerClass[className] = {}
            testScorePerClass[className]['TotalAppearences'] = 0
            testScorePerClass[className]['TruePositive'] = 0
            testScorePerClass[className]['TrueNegative'] = 0
            testScorePerClass[className]['FalsePositive'] = 0
            testScorePerClass[className]['FalseNegative'] = 0

        for i in range(len(y_predict_binary)):
            prediction, truth = y_predict_binary[i], y_test_binary[i]
            for classNum, className in self.classDictionary.items():
                y_truth = truth[classNum]
                y_pred = prediction[classNum]

                if (y_truth == True and y_pred == True):
                    testScorePerClass[className]['TotalAppearences'] += 1
                    testScorePerClass[className]['TruePositive'] += 1
                elif (y_truth == False and y_pred == False):
                    testScorePerClass[className]['TrueNegative'] += 1
                elif (y_truth == False and y_pred == True):
                    testScorePerClass[className]['FalsePositive'] += 1
                elif (y_truth == True and y_pred == False):
                    testScorePerClass[className]['TotalAppearences'] += 1
                    testScorePerClass[className]['FalseNegative'] += 1

        print("testScorePerClass. To run: %s.classPredScore.head(4)" % self.modelName)
        for name, results in testScorePerClass.items():
            print("%s - %s" % (name, str(results)))

        self.classPredScore = pd.DataFrame(testScorePerClass)


"""
testClass = OmerSuperModel(X_train_norm, y_train_Catg, X_val_norm, y_val_Catg, X_test_norm, y_test_Catg,
                           name="testClass", classDictionary=classDict, basePath=baseFolder)
testClass.add_convolution_layer(flattenLayer=True, isFirst=True, addPooling=False, filters=16, kernelSize=(3, 3),
                                activation='relu')
testClass.add_dense_layer(isLastLayer=False, units=34, activation='relu')
testClass.add_dense_layer(isLastLayer=True, units=2, activation='relu')
testClass.compile_model(optimizer='SGD', learning_rate=0.01, momentum=0.00, loss='binary_crossentropy',
                        warmup_proportion=0.1, total_steps=10000, min_lr=1e-5, measurment_metric_name='accuracy')
# testClass.summary()
testClass.trainModel(batch_size=128, epochs=2, alsoTestModel=True, saveTempModel=True)
testClass.save_model()
testClass.compare_all_models()  # (['testClass2','OmerTest'])
testClass.compare_json.sort_values(by=["accuracy"], ascending=False).head(10)
"""

# testClass.compare_json.sort_values(by=["accuracy"],ascending=False).head(10)
# testClass.compare_all_models()
# testClass.compare_json.sort_values(by=['accuracy'],ascending=False).head(10)
# testClass = testClass.load_model(X_train, y_train, X_val, y_val, X_test, y_test,classDictionary=classDictionary)
# testClass.testModel()
# testClass.save_model()
# testClass.display_input_data()

# testClass.evaluate_save_model()
# testClass.compare_all_models()
# testClass.showTrainingHistory()
# listOfImagesAndValues = testClass.testModel()
# testClass.evaluate_save_model()


"""testClass= None
testClass = OmerSuperModel(X_train_norm, y_train_Catg, X_val_norm, y_val_Catg, X_test_norm, y_test_Catg,name="testClass",classDictionary=classDict,basePath=baseFolder)
testClass.add_convolution_layer(flattenLayer=True,isFirst=True, addPooling=False,filters=16,kernelSize=(3,3),activation='relu')
testClass.add_dense_layer(isLastLayer=False,units=34, activation='relu')
testClass.add_dense_layer(isLastLayer=True,units=2, activation='relu')
testClass.compile_model(optimizer='SGD',learning_rate=0.01,momentum=0.00,loss='binary_crossentropy',warmup_proportion=0.1,total_steps=10000,min_lr=1e-5,measurment_metric_name='accuracy')
testClass.trainModel(batch_size=128,epochs=2,alsoTestModel=True,saveTempModel=True,plotResults=False)
#testClass.classificationReport_MultiLabel()
testClass.classPredScore.head(4)"""

"""
testClass = OmerSuperModel(X_train_norm, y_train_Catg, X_val_norm, y_val_Catg, X_test_norm, y_test_Catg,name="testClass",classDictionary=classDict,basePath=baseFolder)
testClass.add_convolution_layer(flattenLayer=True,isFirst=True, addPooling=False,filters=16,kernelSize=(3,3),activation='relu')
testClass.add_dense_layer(isLastLayer=False,units=34, activation='relu')
testClass.add_dense_layer(isLastLayer=True,units=2, activation='relu')
testClass.compile_model(optimizer='SGD',learning_rate=0.01,momentum=0.00,loss='binary_crossentropy',warmup_proportion=0.1,total_steps=10000,min_lr=1e-5,measurment_metric_name='accuracy')
#testClass.summary()
testClass.trainModel(batch_size=128,epochs=2,alsoTestModel=True,saveTempModel=True)
testClass.save_model()
#testClass.compare_all_models()#(['testClass2','OmerTest'])
#testClass.compare_json.sort_values(by=["accuracy"],ascending=False).head(10)
testClass.test_real_images(realImagePath='/content/drive/My Drive/Facebook tagged photos', threshold=0.5,batchSize=10,
                           modelImgShape=testClass.image_shape[0],showOnlyMatches=False,showSFW=False)"""

# testClass.compare_json.sort_values(by=["accuracy"],ascending=False).head(10)
# testClass.compare_all_models()
# testClass.compare_json.sort_values(by=['accuracy'],ascending=False).head(10)
# testClass = testClass.load_model(X_train, y_train, X_val, y_val, X_test, y_test,classDictionary=classDictionary)
# testClass.testModel()
# testClass.save_model()
# testClass.display_input_data()

# testClass.evaluate_save_model()
# testClass.compare_all_models()
# testClass.showTrainingHistory()
# listOfImagesAndValues = testClass.testModel()
# testClass.evaluate_save_model()

""" Snippet to quickly load an image and convert if from BGR to either RGB or Grayscale"""


def load_img_omer(path, isGray=False, showImg=False):
    img = cv2.imread(path)

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


def load_dataset(x_fileName, y_fileName, classDict, sampleDataset=True):
    # load dataset
    npx = np.load(x_fileName)
    npy = np.load(y_fileName)
    X, X_test, y, y_test = model_selection.train_test_split(npx, npy, test_size=0.2, random_state=13)
    # garbage collection or run out of RAM
    npx = None
    npy = None
    # gc.collect()
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.2, random_state=17)
    # one hot encode target values
    y_train_catg = to_categorical(y_train)
    y_val_catg = to_categorical(y_val)
    y_test_catg = to_categorical(y_test)
    print(X_train.shape, X_val.shape, X_test.shape)
    print(y_train.shape, y_val.shape, y_test.shape)

    # DIsplay samples
    if sampleDataset:
        sampleImgs = list()
        sampleTitles = list()
        for i in range(15):
            randInt = random.randint(0, len(X_train))
            img = X_train[randInt]
            label = y_train[randInt]
            title = "Img %s - %s (%s)" % (randInt, classDict[label], label)
            sampleImgs.append(img)
            sampleTitles.append(title)
        display_images_in_plot(sampleImgs, sampleTitles)

    # normalize
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    return X_train, y_train_catg, X_val, y_val_catg, X_test, y_test_catg


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

    print(classDict)
    print(reverseClassDict)
    return classDict, reverseClassDict


if __name__ == '__main__':
    classDict, reverseClassDict = loadClassDict(
        "omerprojects/NudityDetector/Classes/ClassDictionary.json")  # /content/drive/My Drive/Harvard HW/Final Project/Classes/ClassDictionary.json
