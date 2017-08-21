#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 04:32:20 2017

@author: sclegg
"""

import numpy as np
import random
import glob
import time
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import lfunctions as lf
import cv2
from cv2 import ml
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split


# Calculate the accuracy of the class predictions given the
# true class values
def accuracy(true, pred):
    accuracy = np.zeros_like(true)
    accuracy[true==pred]=1
    return sum(accuracy)/len(accuracy)

# Read in cars and notcars
images = glob.glob('./training set smallest/**/*.jpeg',recursive=True)
#images = glob.glob('./training set/**/*.png',recursive=True)
cars = []
notcars = []
for image in images:
    if 'notcar' in image:
        notcars.append(image)
    else:
        cars.append(image)

# Randome shuffle cars and notcars to minimuze the impact of
# sequential video frames.
random.shuffle(cars)
random.shuffle(notcars)

imageSize = cv2.imread(cars[0]).shape # Get training image size (use for HOG)

# Reduce positive and negative image set to size N
N = len(cars)
cars = cars[0:N]
notcars = notcars[0:N]

# Define the training parameters
colorSpace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
colorChannel = 'ALL' # Can be 0, 1, 2, or "ALL"

# Define spatial feature extraction parameters
spatialSize = (32, 32) # Spatial feature size
spatialFeat = False # Use spatial features

# Define color binning and extraction parameters
histBins = 32    # Number of color histogram bins
histFeat = False # Use color features

# Define HOG parameters
winSize = (imageSize[0], imageSize[1]) # HOG window size
blockSize = tuple(a // 4 for a in winSize) # HOG block size
blockStride = tuple(a // 2 for a in blockSize) # HOG block stride
cellSize = blockStride # HOG cell size
nbins = 9 # HOG number of HOG vectors per cell
derivAperture = 1
winSigma = 1. # HOG gaussian smoothing window
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01 # HOG L2-hys threshold
gammaCorrection = False # HOG Use gamma correction
nlevels = 64 # HOG Maximum number of detection window increases 
hogFeat = True # Use HOG features
dictionary = {'accuracy':[],'blockSize':[],'blockStride':[],'cellSize':[],'SVM':[]}
count = 0
for block in [1, 2, 4, 8, 16]:
    for stride in [1, 2, 4, 8, 16]:
        if stride < block:
            print("skipping stride {}".format((block,stride)))
            continue
        for cell in [1, 2, 4, 8, 16]:
            if cell < block:
                print("skipping cell {}".format((block,cell)))
                continue
            blockSize = tuple(a // block for a in winSize) # HOG block size
            blockStride = tuple(a // stride for a in winSize)
            cellSize = tuple(a // cell for a in winSize)
            # Initialzie HOG descriptor
            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                                    derivAperture, winSigma, histogramNormType,
                                    L2HysThreshold, gammaCorrection, nlevels)
            
            print("Initialized HOG descripter")
            # Extract the features for the positive images (cars)
            carFeatures = lf.extractFeatures(cars, hog, hogFeat=hogFeat, 
                                             colorSpace=colorSpace,
                                             colorChannel=colorChannel,
                                             spatialSize=spatialSize,
                                             spatialFeat=spatialFeat,
                                             histBins=histBins,
                                             histFeat=histFeat)
            print("Extracted car {} features".format(len(cars)))
            
            # Extract the features for the negitive images (notcar)
            notcarFeatures = lf.extractFeatures(notcars, hog, hogFeat=hogFeat, 
                                                colorSpace=colorSpace,
                                                colorChannel=colorChannel,
                                                spatialSize=spatialSize,
                                                spatialFeat=spatialFeat,
                                                histBins=histBins,
                                                histFeat=histFeat)
            
            print("Extracted none car {} features".format(len(notcars)))
            print()
            
            # Vertically stack the car and not car feature sets
            # Also convert to float64
            X = np.vstack((carFeatures, notcarFeatures)).astype(np.float32)
            
            # Apply StandardScaler to fit a per-column scaler
            # The purpose is to normalize the the training dataset 
            xScaler = StandardScaler().fit(X)
            # Apply the scaler to X
            scaledX = xScaler.transform(X)
            
            # Define the labels vector
            y = np.hstack((np.ones(len(carFeatures)).astype('int'), np.zeros(len(notcarFeatures)).astype('int')))
            
            # Split up data into randomized training and test sets
            # Hold 20% of the data aside for testing
            #rand_state = np.random.randint(0, 100)
            rand_state = 86 # fix the random state to insure that splits are always the same
            xTrain, xTest, yTrain, yTest = train_test_split(
                scaledX, y, test_size=0.2, random_state=rand_state)
            
            print("Using: {} color space".format(colorSpace))
            print("       color channel {}".format(colorChannel))
            print()
            print("Using spatial features: {}".format(spatialFeat))
            print("      spatial feature size {}".format(spatialSize))
            print()
            print("Using color features: {}".format(histFeat))
            print("      color bin size {}".format(histBins))
            print()
            print("Using HOG features: {}".format(hogFeat))
            print("      {} HOG window size".format(winSize))
            print("      {} HOG block size".format(blockSize))
            print("      {} HOG block stride".format(blockStride))
            print("      {} HOG cell size".format(cellSize))
            print("      {} HOG orientations".format(nbins))
            print()
            print("Feature vector length: {}".format(len(xTrain[0])))
            print()
            # Initialize SVM
            SVM = ml.SVM_create()
            SVM.setType(ml.SVM_C_SVC)
            
            # Check the training time for the SVC
            t1=time.time()
            kernel = ml.SVM_LINEAR
            SVM.setKernel(kernel)
            C = 10.
            SVM.setC(C)
            SVM.train(xTrain, cv2.ml.ROW_SAMPLE, yTrain)
            yPred = SVM.predict(xTest)[1].ravel().astype('int')
            acc = accuracy(yTest, yPred)
            print("Trained for Linear kernel with C={}. Accuracy is {:6.3f}".format(C,acc))
            dictionary['accuracy'].append(acc)
            dictionary['blockSize'].append(blockSize)
            dictionary['blockStride'].append(blockStride)
            dictionary['cellSize'].append(cellSize)
            dictionary['SVM'].append(SVM)
print()
print()
acc = dictionary['accuracy']
bestAccuracy = max(acc)
index = acc.index(bestAccuracy)
blockSize = dictionary['blockSize'][index]
blockStride = dictionary['blockStride'][index]
cellSize = dictionary['cellSize'][index]
SVM = dictionary['SVM'][index]
print("Best accuracy: {:6.2f}".format(bestAccuracy))
print("   With parameters:")
print("         blockSize:{}".format(blockSize))
print("       blockStride:{}".format(blockStride))
print("          cellSize:{}".format(cellSize))
t3=time.time()

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                        derivAperture, winSigma, histogramNormType,
                        L2HysThreshold, gammaCorrection, nlevels)

print("Initialized HOG descripter")
# Extract the features for the positive images (cars)
carFeatures = lf.extractFeatures(cars, hog, hogFeat=hogFeat, 
                                 colorSpace=colorSpace,
                                 colorChannel=colorChannel,
                                 spatialSize=spatialSize,
                                 spatialFeat=spatialFeat,
                                 histBins=histBins,
                                 histFeat=histFeat)
print("Extracted car {} features".format(len(cars)))

# Extract the features for the negitive images (notcar)
notcarFeatures = lf.extractFeatures(notcars, hog, hogFeat=hogFeat, 
                                    colorSpace=colorSpace,
                                    colorChannel=colorChannel,
                                    spatialSize=spatialSize,
                                    spatialFeat=spatialFeat,
                                    histBins=histBins,
                                    histFeat=histFeat)

print("Extracted none car {} features".format(len(notcars)))
print()

# Vertically stack the car and not car feature sets
# Also convert to float64
X = np.vstack((carFeatures, notcarFeatures)).astype(np.float32)

# Apply StandardScaler to fit a per-column scaler
# The purpose is to normalize the the training dataset 
xScaler = StandardScaler().fit(X)
# Apply the scaler to X
scaledX = xScaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(carFeatures)).astype('int'), np.zeros(len(notcarFeatures)).astype('int')))

# Split up data into randomized training and test sets
# Hold 20% of the data aside for testing
rand_state = 86 # fix the random state to insure that splits are always the same
xTrain, xTest, yTrain, yTest = train_test_split(
    scaledX, y, test_size=0.2, random_state=rand_state)
yTrue, yPred = yTest, SVM.predict(xTest)[1].ravel().astype('int32')
t4=time.time()
print("{} Seconds for SVC predictions...".format(round(t4-t3, 2)))

print(classification_report(yTest, yPred))
print()
