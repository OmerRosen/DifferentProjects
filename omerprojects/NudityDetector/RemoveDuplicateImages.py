# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os

def dhash(image, hashSize=8):
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash and return it
    output = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    return output


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset")
# ap.add_argument("-r", "--remove", type=int, default=-1,
# 	help="whether or not duplicates should be removed (i.e., dry run)")
# args = vars(ap.parse_args())

#args = {"dataset":r"C:\Users\omerro\Google Drive\Data Science Projects\OmerPortal\omerprojects\NudityDetector\Classes",
args = {"dataset":r"C:\Users\omerro\AppData\Local\Temp",
        "remove":1,
        }

# grab the paths to all images in our input dataset directory and
# then initialize our hashes dictionary
print("[INFO] computing image hashes...")
imagePaths = list(paths.list_images(args["dataset"]))
print("Total of %s image pathes colected\n\n"%(len(imagePaths)))
hashes = {}
# loop over our image paths
for imagePath in imagePaths:
	# load the input image and compute the hash
	image = cv2.imread(imagePath)
	h = dhash(image)
	# grab all image paths with that hash, add the current image
	# path to it, and store the list back in the hashes dictionary
	p = hashes.get(h, [])
	p.append(imagePath)
	hashes[h] = p

print("Finished mapping. %s unique hashes out of %s images"%(len(hashes),len(imagePaths)))

# loop over the image hashes
duplicateCount=0
for (h, hashedPaths) in hashes.items():
    # check to see if there is more than one image with the same hash
    if len(hashedPaths) > 1:
        duplicateCount+=1
        print("\nFound %s images under hash %s"%(len(hashedPaths),h))
        #check to see if this is a dry run
        if args["remove"] <= 0:
            # initialize a montage to store all images with the same
            # hash
            montage = None
            # loop over all image paths with the same hash
            for p in hashedPaths:
                print(p)
                # load the input image and resize it to a fixed width
                # and heightG
                image = cv2.imread(p)
                image = cv2.resize(image, (150, 150))
                # if our montage is None, initialize it
                if montage is None:
                    montage = image
                # otherwise, horizontally stack the images
                else:
                    montage = np.hstack([montage, image])
            # show the montage for the hash
            print("[INFO] hash: {}".format(h))
            #cv2.imshow("Montage", montage)
            #cv2.waitKey(0)

# otherwise, we'll be removing the duplicate images
        else:
            # loop over all image paths with the same hash *except*
            # for the first image in the list (since we want to keep
            # one, and only one, of the duplicate images)
            for p in hashedPaths[1:]:
                os.remove(p)

print("Finished process. Total %s duplication items found"%(duplicateCount))