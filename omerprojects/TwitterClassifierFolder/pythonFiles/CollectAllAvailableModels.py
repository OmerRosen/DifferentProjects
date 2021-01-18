import os
from pathlib import Path
import json
import pandas as pd


def gatherAllAvailableModels(baseFolder):
    modelPath = baseFolder
    listOfFolders = os.listdir(modelPath)

    listOfModels = []

    for folder in listOfFolders:
        projectName = folder
        filePaths = os.listdir(os.path.join(modelPath, folder))
        for file in filePaths:
            if file == 'Model_HyperParameters.txt':
                filePath = os.path.join(modelPath, folder, file)
                with open(filePath) as json_file:
                    model_HyperParameters = json.load(json_file)
                    print("Model name: %s " % file)
                    print("Model_HyperParameters: \n")
                    print(model_HyperParameters)
                    listOfModels.append(model_HyperParameters)
    return listOfModels


if __name__ == "__main__":
    currentFolder = Path(os.getcwd())
    modelPath = os.path.join(currentFolder.parent, 'models')

    gatherAllAvailableModels(baseFolder=modelPath)
