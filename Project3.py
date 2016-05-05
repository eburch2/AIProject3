
# coding: utf-8

# In[185]:

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from os import getcwd
from os.path import join
import sys

def main():
    
    testFileName = "test.jpg"
    
	#for hardcoding files
    #args = ["Project3.py", getcwd() + "\\" + testFileName]
    args = sys.argv
    
    testFileDirectory = getcwd() + "\\" + testFileName
    
    if (len(args) != 2):
        print("Please run the program with a single command-line argument including a filepath to an image")
        print("ex: Project3.py TestFile.jpg")
        print("Exiting")
        exit()
    else:
        testFileName = sys.argv[1]
        
    # Set this to false to re-train SVM. Otherwise true will load the pickled SVM
    isSVMTrained = True
    
    if (testFileName == "TEACH_SVM"):
        isSVMTrained = False    
    
    FOLDER_NAMES = "01", "02", "03", "04", "05"
    CLASSIFICATIONS = "Smile", "Hat", "Hashtag", "Heart", "Dollar"
    TESTING_FOLDER = "TESTING"
    
    numRight = 0
    numWrong = 0
    numTotal = 0
        
    clf = np.array([])    
        
    allImages = np.array([])
    allClassifications = np.array(CLASSIFICATIONS)
    
    # Loads a pickled SVM model, otherwise creates a new model and then saves it
    if (isSVMTrained == True):
        clf = loadSVM()
    else:        
        allImages, allClassifications = loadAllImages()
        clf = teachSVM(allImages, allClassifications)
        saveSVM(clf)
		print("SVM Trained. Exiting.")
		exit()
        
    # Use these for validation
    #folds = allClassifications.size
    #folds = 10
    #allImages, allClassifications = loadAllImages()
    #validate(allImages, allClassifications, folds)     
    
    # Use for individual testing of a file
    '''
    actualLabel = "Dollar"
    predictedLabel = classifyImage(testFileDirectory, clf)[0]
    if(actualLabel == predictedLabel):
        print("Correct Match! Successfully labeled: " + predictedLabel)
    else:
        print("Incorrect Match! Actual: " + actualLabel + ", predicted: " + predictedLabel)
        
    print("The image is predicted to be: " + predictedLabel)      
    '''
    
    # Use this for viewing pleasures
    '''    
    testImageFile = loadImage(testFileDirectory)
    plt.imshow(testImageFile)
    plt.show
    '''
    


# In[186]:

from sklearn import svm
from sklearn import preprocessing
    
def classifyImage(imageDirectory, clf):

    label = []
    testImageFile = loadImage(imageDirectory)
    
    if (len(testImageFile) == 0):
        print("FILE NOT FOUND. EXITING.")
    else:
        print("Loaded")
        testImageFile = np.ravel(testImageFile)
        #testImageFile = preprocessing.scale(testImageFile)
        #testImageFile = preprocessing.normalize(testImageFile, norm='l2', axis=1, copy=False)
        
        label = clf.predict(testImageFile).tolist()        
    return label  


# In[187]:

from sklearn.externals import joblib

def saveSVM(clf):
    joblib.dump(clf, 'dumpSVM.pkl') 

def loadSVM():
    
    clf = joblib.load('dumpSVM.pkl')     
    return clf    


# In[188]:

FOLDER_NAMES = "01", "02", "03", "04", "05"
CLASSIFICATIONS = "Smile", "Hat", "Hashtag", "Heart", "Dollar"
TRAINING_FOLDER = "TRAINING"

from glob import glob
from os import getcwd
from os.path import join
from sklearn import svm
from sklearn import preprocessing

def teachSVM(trainingSet, classifications):
        
    #print(trainingSet.shape)
    #print(classifications.shape)
    
    X = trainingSet
    y = classifications
    
    #X = preprocessing.scale(X)
    #X = preprocessing.normalize(X)
    #print(X)
    clf = svm.LinearSVC()
    clf.fit(X, y)
    
    #xFile.close() 
    #yFile.close()     
    return clf


# In[189]:

# Repeated for referencing
#FOLDER_NAMES = "01", "02", "03", "04", "05"
#CLASSIFICATIONS = "Smile", "Hat", "Hashtag", "Heart", "Dollar"
#TRAINING_FOLDER = "TRAINING"

def loadAllImages():
    
    X = []
    y = []
    
    print("Loading all images in training set. Please wait...")
    
    for folder in FOLDER_NAMES:
        for imageDirectory in glob(join(getcwd(), TRAINING_FOLDER, folder, '*.jpg')):

            imageFile = loadImage(imageDirectory)

            if (len(imageFile) == 0):
                print("FILE NOT FOUND. EXITING.")
                break;
            else:
                #print("Loaded")
                # adds our scaled/normalized vector to X and its classification to y
                newImageFile = np.asarray(imageFile).ravel()
                #print(newImageFile.shape)
                X.append(newImageFile)
                y.append(CLASSIFICATIONS[FOLDER_NAMES.index(folder)])

                #xFile.write(str(imageFile) + "\n")
                #yFile.write(CLASSIFICATIONS[FOLDER_NAMES.index(folder)] + "\n")    

        print("Loading complete from folder: " + folder)
    print("Loading complete")
    
    #X = np.asarray(X).reshape((len(X), 1))
    X = np.asarray(X)
    #print(X.shape)
    y = np.asarray(y)
    #print(y.shape)
        
    return X, y


# In[190]:

import numpy as np
import math as math

def validate(images, classifications, numFolds):
    
    numRight = 0
    numWrong = 0
    numTotal = 0
    
    maxRange = 0
    width = 0    
    
    width = math.floor(classifications.size / numFolds)    
    if (width == 0):
        width = 1
                
    maxRange = math.floor(classifications.size / width)
    
    #print("all images: " + str(images))
    #print(str(width))
    for i in range(0, maxRange):
        
        testStart = i*width
        testStop = (i+1)*width
        
        trainingSetPt1i = images[:testStart]
        trainingSetPt2i = images[testStop:]
        trainingSetPt1c = classifications[:testStart]
        trainingSetPt2c = classifications[testStop:]
        
        trainingSeti = np.concatenate([trainingSetPt1i, trainingSetPt2i])
        trainingSetc = np.concatenate([trainingSetPt1c, trainingSetPt2c])
        trainingSetc = np.ravel(trainingSetc)
            
        #print("training set: " + str(trainingSeti) + ", " + str(trainingSetc))
        #print("training set: " + ", " + str(trainingSetc))
        testingSeti = images[testStart:testStop]
        testingSetc = classifications[testStart:testStop]
        #print("testing set: " + str(testingSeti) + "," + str(testingSetc))
        #print("testing set: " + ", " + str(testingSetc))
        
        #trainingSeti = trainingSeti.reshape(9, 2)
        #trainingSetc = trainingSetc.reshape(trainingSetc.size, 1)
        
        if (trainingSeti.size != 0):
            #print(str(trainingSeti))
            #print(str(trainingSetc))
            clf = teachSVM(trainingSeti, trainingSetc)

            for testNum in range(0, testingSetc.size):

                actualLabel = testingSetc[testNum]
                predictedLabel = clf.predict(testingSeti[testNum]).tolist()[0]

                #print(actualLabel)
                #print(predictedLabel)

                if(actualLabel == predictedLabel):
                    print("Correct Match! Successfully labeled: " + predictedLabel)
                    numRight += 1
                else:
                    print("Damn it! Actual: " + actualLabel + ", predicted: " + predictedLabel)
                    numWrong += 1
                numTotal += 1
        
        
    print("Number of right guesses: " + str(numRight))
    print("Number of wrong guesses: " + str(numWrong))
    print("Total | Accuracy: " + str(numTotal) + " | " + str((numRight / numTotal) * 100) + "%")


# In[203]:

# Formats for the parameters are as shown below
# parentFolder - "TRAINING" or "TESTING"
# folderNum - "01", "02", "03", "04", or "05"
# fileNameWExtension - "01.jpg", "02.jpg", ..., "10.jpg", ... etc

from scipy import misc
from scipy import ndimage

def loadImage(fileLocation):
    
    
    imageFile = np.array([])
    try:
        imageFile = Image.open(fileLocation)
    except:
        print('FILE NOT FOUND IN DIR: ' + fileLocation)
        
    if (imageFile):
        #np.set_printoptions(threshold=np.nan)
        bitmap = np.asarray(imageFile)
        bitmap = threshold(bitmap)
        
        bitmap = ndimage.gaussian_filter(bitmap, sigma = 1.5)
        
        return bitmap
    else:
        return np.array([])


# In[204]:

WHITE = 255
BLACK = 0

def threshold(bitmap):
    bitmap.flags.writeable = True
    newBitMap = bitmap
    avgArray = []
    
    for row in bitmap:
        for pixel in row:
            avgRGB = np.average(pixel)
            avgArray.append(avgRGB)
        
    breakingPoint = np.average(avgArray)
    
    for row in newBitMap:
        for pixel in row:
            
            # Changes the RGB to either black or white depending on threshold/breaking point
            newRGB = 0
            if (np.average(pixel) > breakingPoint):
                newRGB = WHITE
            else:
                newRGB = BLACK
                
            for index in range(0, len(pixel)):
                pixel[index] = newRGB
                
    return newBitMap
    


# In[205]:

main()


# In[ ]:




# In[ ]:



