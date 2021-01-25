from omerprojects.NudityDetector.ImageClassifierClass import OmerSuperModel
import os

baseFolder = '/content/drive/My Drive/Harvard HW/Final Project'
baseModelFolder = os.path.join(baseFolder, 'Models')
ClassFolderPath = os.path.join(baseFolder,'Classes')
ClassDictPath = os.path.join(baseFolder,'Classes/ClassDictionary.json')
print("baseFolder"+baseFolder)
print("baseModelFolder"+baseModelFolder)
print("ClassDictPath"+ClassDictPath)