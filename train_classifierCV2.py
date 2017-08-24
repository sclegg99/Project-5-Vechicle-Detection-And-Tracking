#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:20:42 2017

@author: UDACITY & Scott Clegg
Routine to train an SVM classifier to distinguish between
cars and not cars.  The initial routine is based on lesson scripts
provided UDACITY for the ND Car term 1.

The script has been modified to use OpenCV HOG and SVM methods.
"""
import numpy as np
import random
import glob
import time
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import findCarFunctions as fc
import cv2
from cv2 import ml
import pickle
from sklearn.model_selection import train_test_split


# Calculate the accuracy of the class predictions given the
# true class values
def accuracy(true, pred):
    accuracy = np.zeros_like(true)
    accuracy[true==pred]=1
    return sum(accuracy)/len(accuracy)

# Read in cars and notcars
#images = glob.glob('./training set smallest/**/*.jpeg',recursive=True)
images = glob.glob('./training set/**/*.png',recursive=True)
cars = []
notcars = []
for image in images:
    if 'notcar' in image:
        notcars.append(image)
    else:
        cars.append(image)

# Randome shuffle cars and notcars to minimuze the impact of
# sequential video frames.
random.seed(9032)
for i in range(5):
    random.shuffle(cars)
    random.shuffle(notcars)

imageSize = cv2.imread(cars[0]).shape # Get training image size (use for HOG)

# Reduce positive and negative image set to size N
N = len(cars)
#N = 100
cars = cars[0:N]
notcars = notcars[0:N]

# Define the training parameters
colorSpace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
colorChannel = 'ALL' # Can be 0, 1, 2, or "ALL"

# Define spatial feature extraction parameters
spatialSize = (16, 16) # Spatial feature size
spatialFeat = True # Use spatial features

# Define color binning and extraction parameters
histBins = 32    # Number of color histogram bins
histFeat = True # Use color features

# Define HOG parameters
winSize = (imageSize[0], imageSize[1]) # HOG window size
blockSize = ( 16, 16) # Optimial HOG block size
blockStride = ( 4, 4) # Optimal HOG block stride
cellSize = (4, 4) # Optimal HOG cell size
nbins = 11 # HOG number of HOG vectors per cell
derivAperture = 1
winSigma = -1 # HOG gaussian smoothing window (use sigma=1; otherwise too much smoothing)
histogramNormType = 0
L2HysThreshold = .2 # HOG L2-hys threshold
gammaCorrection = True # HOG Use gamma correction
nlevels = 64 # HOG Maximum number of detection window increases
signed_gradient = False
hogFeat = True # Use HOG features

# Initialzie HOG descriptor
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                        derivAperture, winSigma, histogramNormType,
                        L2HysThreshold, gammaCorrection, nlevels,
                        signed_gradient)
print("Initialized HOG descripter")

# Extract the features for the positive images (cars)
carFeatures = fc.extractFeatures(cars, hog, hogFeat=hogFeat, 
                                 colorSpace=colorSpace,
                                 colorChannel=colorChannel,
                                 spatialSize=spatialSize,
                                 spatialFeat=spatialFeat,
                                 histBins=histBins,
                                 histFeat=histFeat)
print("Extracted car {} features".format(len(cars)))

# Extract the features for the negitive images (notcar)
notcarFeatures = fc.extractFeatures(notcars, hog, hogFeat=hogFeat, 
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
y = np.hstack((np.ones(len(carFeatures)),
               np.zeros(len(notcarFeatures)))).astype('int32')

# Split up data into randomized training and test sets
#rand_state = np.random.randint(0, 100)
rand_state = 9331
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

# Check the training time for the SVC
t1=time.time()

print("# Tuning hyper-parameters")
print()

dictionary = {'accuracy':[],'kernel':[],'C':[],'gamma':[],'SVM':[]}

gamma = 0
for C in [.000005, .00001, .000015, .00002, .000025]:
    # Initialize SVM
    SVM = ml.SVM_create()
    SVM.setType(ml.SVM_C_SVC)
    # Tune SVM linear function model
    kernel = ml.SVM_LINEAR
    SVM.setKernel(kernel)
    SVM.setC(C)
    
    # Train the SVM
    SVM.train(xTrain, cv2.ml.ROW_SAMPLE, yTrain)
    
    # Predict the test values
    yPred = SVM.predict(xTest)[1].ravel().astype('int')
    
    # Calucalate the accuracy of the predictions
    acc = accuracy(yTest, yPred)
    print("Trained for Linear kernel with C={}. Accuracy is {:6.3f}".format(C,acc))
    dictionary['accuracy'].append(acc)
    dictionary['kernel'].append(kernel)
    dictionary['C'].append(C)
    dictionary['gamma'].append(gamma)
    dictionary['SVM'].append(SVM)
"""
# Don't use RBF because it consumes too much time for predictions
for C in [.1, 1, 10, 100]:
    for gamma in [.0001, .001, .01]:
        # Initialize SVM
        SVM = ml.SVM_create()
        SVM.setType(ml.SVM_C_SVC)
        # Tune SVM radial bias function model
        kernel = ml.SVM_RBF
        SVM.setKernel(kernel)
        SVM.setC(C)
        SVM.setGamma(gamma)
        
        # Train the SVM
        SVM.train(xTrain, cv2.ml.ROW_SAMPLE, yTrain)
        
        # Calucalate the accuracy of the predictions
        yPred = SVM.predict(xTest)[1].ravel().astype('int')
        acc = accuracy(yTest, yPred)
        print("Trained for Linear kernel with C={} and gamma={}. Accuracy is {:6.3f}".format(C,gamma,acc))
        dictionary['accuracy'].append(acc)
        dictionary['kernel'].append(kernel)
        dictionary['C'].append(C)
        dictionary['gamma'].append(gamma)
        dictionary['SVM'].append(SVM)
"""
acc = dictionary['accuracy']
bestAccuracy = max(acc)
index = acc.index(bestAccuracy)
kernel = dictionary['kernel'][index]
C = dictionary['C'][index]
gamma = dictionary['gamma'][index]
SVM = dictionary['SVM'][index]
print()
print("Best parameters:")
print("       accuracy: {}".format(bestAccuracy))
print("         kernel: {}".format(kernel))
print("              C: {}".format(C))
print("          gamma: {}".format(gamma))
print()

# Predict the class (car vs not car) for the test set and report
# the accuracy of the text set.
t3=time.time()
yTrue, yPred = yTest, SVM.predict(xTest)[1].ravel().astype('int32')
t4=time.time()
print("{:6.3f} Seconds for {} SVC predictions...".format(t4-t3, N))

print(classification_report(yTest, yPred))
print()

# Check the score of the SVC
print("Test Accuracy of SVC = {}".format(round(accuracy(yTest, yPred), 4)))

# Save optimal training data to a pickle file
SVMfile = 'SVM.xml'
SVM.save(SVMfile)
support_vectors = SVM.getSupportVectors()
training_dat = {'clf': SVMfile,
                'C': C,
                'support vectors': support_vectors,
                'colorSpace': colorSpace,
                'colorChannel': colorChannel,
                'spatialSize': spatialSize,
                'spatialFeat': spatialFeat,
                'histBins': histBins,
                'histFeat': histFeat,
                'winSize': winSize,
                'blockSize': blockSize,
                'blockStride': blockStride,
                'cellSize': cellSize,
                'nbins': nbins,
                'derivAperture': derivAperture,
                'winSigma': winSigma,
                'histogramNormType': histogramNormType,
                'L2HysThreshold': L2HysThreshold,
                'gammaCorrection': gammaCorrection,
                'nlevels': nlevels,
                'signed_gradient' : signed_gradient,
                'hogFeat': hogFeat,
                'xScaler': xScaler}
with open('svcPickle.p', 'wb' ) as f:
    pickle.dump(training_dat, f)
f.close