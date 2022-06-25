#%%
# design for multi-examples (2017_6_6)
#%%

import os
import numpy as np # Numpy for general purpose processing
import SimpleITK as sitk # SimpleITK to load images
from sklearn.metrics import f1_score

class CompareThePreandTruth(object):
    """ class for multi atlas segmentation base on elastix """
    
    def __init__(self, predictListdir, groundTruthListdir):
        self.predictListdir = predictListdir
        self.groundTruthListdir = groundTruthListdir
        
    def readPredictImagetoList(self):
        """
        read predicted image's dirs to list
        """
        self.predictList = []
        with open(self.predictListdir) as f:
            self.predictList = f.read().splitlines()
        self.predictList.sort()        
        return self.predictList
    
    def readgroundTruthtoList(self):
        """
        read groundTruth image's dirs to list
        """
        self.groundTruthList = []
        with open(self.groundTruthListdir) as f:
            self.groundTruthList = f.read().splitlines()
        self.groundTruthList.sort()       
        return self.groundTruthList

    def thresholdModification(self, InputImageList, result_dir, threshold = 10**(-2)):
        outputlist = []
        N = len(InputImageList)
        for i in range(N):
            image = sitk.ReadImage(InputImageList[i])
            image_array = sitk.GetArrayFromImage(image) # get numpy array

            image_array[image_array > threshold] = np.uint8(1)
            image_array[image_array <= threshold] = np.uint8(0)

            img = sitk.GetImageFromArray(image_array)
            img.SetOrigin(image.GetOrigin())
            img.SetSpacing(image.GetSpacing())
            img.SetDirection(image.GetDirection())

            name, ext = os.path.splitext(InputImageList[i])
            baseName = os.path.basename(name)
            fn = result_dir + '/thresholdMod_' + baseName + '.nrrd'
            outputlist.append(fn)
            sitk.WriteImage(img,fn)
        outputlist.sort()
        return outputlist

    def predictModification(self, result_dir, threshold):
        self.predictList = self.thresholdModification(self.predictList,\
         result_dir, threshold)
        self.predictList.sort()
        return self.predictList
        
    def groundTruthModification(self, result_dir, threshold):
        self.groundTruthList = self.thresholdModification(self.groundTruthList,\
         result_dir, threshold)
        self.groundTruthList.sort()
        return self.groundTruthList
        
    def diceScoreStatistics(self):
        
        if len(self.predictList)!= len(self.groundTruthList):
            raise ValueError('the num of predicted images\
            should match that of the ground truth iamges')
        self.listLength = len(self.predictList)
        
        diceScore = np.zeros((self.listLength,))    
        for i in range(self.listLength):
            
            ImgroundTruth = sitk.ReadImage(self.groundTruthList[i])
            TmGT_array = sitk.GetArrayFromImage(ImgroundTruth)
            y_true = np.reshape(TmGT_array,-1)            

            ImPred = sitk.ReadImage(self.predictList[i])
            ImPred_array = sitk.GetArrayFromImage(ImPred)
            y_pred = np.reshape(ImPred_array,-1)
            
            diceScore[i] = f1_score(y_true, y_pred)

        dice_Statistics = {}
        dice_Statistics['mean'] = np.mean(diceScore)
        dice_Statistics['std'] = np.std(diceScore)
        dice_Statistics['max'] = np.amax(diceScore)
        dice_Statistics['min'] = np.amin(diceScore)
        print diceScore
        print dice_Statistics