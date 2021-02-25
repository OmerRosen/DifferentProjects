from omerprojects.NudityDetector.MakePredictionUsingWinningModel import takeImagePath_ReturnPredictions
import sys
import os
import requests
from urllib.request import urlretrieve
import time
from datetime import datetime, timedelta
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from omerprojects.__init__ import app
from PIL import Image
import base64
import shutil


search_term = "bathing suit model male" # tomatoes, bell peppers, beetroot
# Setup base download folder
downloads = os.path.join(app.config['BASE_FOLDER'],"NudityDetector/Crawler/downloads")
num_images_requested=200
batchSize = 50
# Each scrolls provides 400 image approximately
number_of_scrolls = int(num_images_requested / 400) + 1

#Path of winning model for prediction
winningModel_AbsPath =  r"C:/Users/omerro/Google Drive/Data Science Projects/OmerPortal/omerprojects/NudityDetector/Models/ModelOutput - NudiyDetector_Draft2/NudiyDetector_Draft2 - Accuracy 0.6.hdf5"

# Imamge link store
imgs_urls = set()

img_dir = os.path.join(downloads,search_term)
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
# Temp download dir: C:\Users\omerro\AppData\Local\Temp

# Using Chrome
chrome_options = webdriver.ChromeOptions()
print(chrome_options.arguments)
#prefs = {"profile.managed_default_content_settings.images": 2}
#chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('--ignore-ssl-errors')
#chrome_options.add_argument('headless')

try:
    # Browser
    ChromePath = os.path.join(app.config['BASE_FOLDER'],r"NudityDetector/Crawler/driver/chromedriver.exe")
    browser = webdriver.Chrome(ChromeDriverManager().install()) #(ChromePath, chrome_options=chrome_options)

    # Main search
    browser.get('https://www.google.com/search?q='+search_term)

    # Get all Google links, filter our "Image" tab
    browserTitleElements = browser.find_elements_by_xpath('//a[contains(@class, "")]')
    imageLink = ''
    for titleBtn in browserTitleElements:
        #print(titleBtn.text)
        if titleBtn.text == "Images":
            print(titleBtn.get_attribute("href"))
            images_link = titleBtn#.get_attribute("href")
            break


    # Wait
    time.sleep(5)

    # Go to images
    # images_link = imagePageLink #images_links[0]
    images_link.click()
    time.sleep(3)

    listOfImageSavedPaths = []


    #Collect element from browser based on last location:
    numOfResults = 0

    # while len(listOfImageSavedPaths)<num_images_requested:
    for pageLoadNum in range(number_of_scrolls):
        for scrollDownNum in range(10):

            # multiple scrolls needed to show all 400 images
            browser.execute_script("window.scrollBy(0, 1000000)")
            time.sleep(2)

        listOfImages = []
        listOfImageTempPaths = []

        url_elements = browser.find_elements_by_xpath('//img')
        browserElementLen = len(url_elements)
        url_elements = url_elements[numOfResults:]
        print("Downloading elements %s to %s"%(numOfResults,browserElementLen))

        listOfImagePaths_ForBatch = []

        for i,url_element in enumerate(url_elements):
            try:
                img_url = url_element.get_attribute('src')
            except Exception as e:
                print("Couldn't find src for element %s: %s"%(i,url_element.text))
                print(e)

            if not img_url in listOfImageTempPaths and img_url is not None:
                listOfImageTempPaths.append(img_url)
                imgTempPath = urlretrieve(img_url)[0]
                imgTempPath.replace('\\', '/')

                os.rename(imgTempPath, imgTempPath + '.jpg')
                listOfImagePaths_ForBatch.append(imgTempPath + '.jpg')
            elif img_url in listOfImageTempPaths:
                print("Already collected image. %s" % url_element.text)
            else: pass

        print('Extracted %s valid image urls out of %s elements' % (len(listOfImagePaths_ForBatch), i))
        numOfResults = len(url_elements)

        imageProbailitiesList = takeImagePath_ReturnPredictions(imagesPathList=listOfImagePaths_ForBatch, requestedModelAbsPath=winningModel_AbsPath)

        for imgName, classes in imageProbailitiesList.items():
            fileDateStamp = datetime.today()
            fileName = "%s_%s_Classes" % (fileDateStamp.strftime("%Y-%m-%d_%H-%M-%S"), imgName)
            filePath = downloads
            for className, prob in classes['Classifications'].items():
                if prob > 0.5:
                    fileName += "_" + className
                    filePath = os.path.join(filePath, className)
                    if not os.path.exists(filePath):
                        os.makedirs(filePath)
            if filePath==downloads: #If no class was recognized, used folder None
                filePath = os.path.join(filePath, "NoClass")
                if not os.path.exists(filePath):
                    os.makedirs(filePath)
            finalPathName = os.path.join(filePath, fileName + '.jpg')
            # print(finalPathName)
            try:
                # shutil.copy(classes['imgPath'], finalPathName)
                os.rename(classes['imgPath'], classes['imgPath'].replace(imgName,fileName))
                listOfImageSavedPaths.append(finalPathName)
            except Exception as e:
                print("Couldn't rename file %s to %s"%(classes['imgPath'], finalPathName))

        print(imageProbailitiesList)

        # to load next 400 images
        time.sleep(5)
        # try to find show more results bottom
        if len(listOfImageSavedPaths)<num_images_requested and len(listOfImagePaths_ForBatch)>0:
            try:
                # if found click to load more image
                browser.find_element_by_xpath("//input[@value='Show more results']").click()
            except Exception as e:
                # if not exit
                print("End of page")
                break
        elif len(listOfImageSavedPaths)<num_images_requested:
            print("Sufficient number of images was reached: %s"%num_images_requested)
            break
        elif len(listOfImagePaths_ForBatch)>0:
            print("No more available images to download")
            break

    print('Found image urls',len(imgs_urls))

    # Wait 5 seconds
    time.sleep(5)

    # Quit the browser
    browser.quit()

except Exception as e:
    print("Error while running browser")
    print(e)
    browser.quit()

# Save the images

#
# for _ in range(number_of_scrolls):
#     for __ in range(10):
#         # multiple scrolls needed to show all 400 images
#         browser.execute_script("window.scrollBy(0, 1000000)")
#         time.sleep(2)
#     # to load next 400 images
#     time.sleep(5)
#     # try to find show more results bottom
#     try:
#         # if found click to load more image
#         browser.find_element_by_xpath("//input[@value='Show more results']").click()
#     except Exception as e:
#         # if not exit
#         print("End of page")
#         break
#
# # Find the thumbnail images
# thumbnails = browser.find_elements_by_xpath('//a[@class="wXeWr islib nfEiy mM5pbd"]')
# # loop over the thumbs to retrive the links
# count=0
# for thumbnail in thumbnails:
#     print(thumbnail)
#     count += 1
#     # check if reached the request number of links
#     if len(imgs_urls) >= num_images_requested:
#         break
#     try:
#         thumbnail.click()
#         time.sleep(2)
#     except Exception as error:
#         print("Error clicking one thumbnail : ", error)
#     # Find the image url
#     url_elements = browser.find_elements_by_xpath('//img[@class="n3VNCb"]')
#     # check for the correct url
#
#     for url_element in url_elements:
#
#         try:
#             url = url_element.get_attribute('src')
#             try:
#
#                 with requests.get(url, stream=True) as r:
#                     r.raise_for_status()
#                     file_path = os.path.join(img_dir, '%s - %s.jpg'%(associatedClass,count))
#                     print("Saving image %s"%(file_path))
#                     with open(file_path, 'wb') as f:
#                         for chunk in r.iter_content(chunk_size=8192):
#                             f.write(chunk)
#                 time.sleep(0.3)
#             except Exception as e:
#                 print("Error in url:", url)
#                 print(e)
#                 continue
#         except e:
#             print("Error getting one url")
#         if url.startswith('http') and not url.startswith('https://encrypted-tbn0.gstatic.com'):
#             imgs_urls.add(url)

# print('Found image urls',len(imgs_urls))
#
# # Wait 5 seconds
# time.sleep(5)
#
# # Quit the browser
# browser.quit()
#
# # Save the images

