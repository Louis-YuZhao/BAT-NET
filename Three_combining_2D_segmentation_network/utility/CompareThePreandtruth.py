import os
import numpy as np # Numpy for general purpose processing
import SimpleITK as sitk # SimpleITK to load images
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from skimage.metrics import hausdorff_distance

def average_volume_difference(truth, prediction):
    return np.abs(np.sum(prediction,dtype=np.int32) - np.sum(truth, dtype=np.int32))/ np.sum(truth)


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
        #print(self.predictList)
        return self.predictList
    
    def readgroundTruthtoList(self):
        """
        read groundTruth image's dirs to list
        """
        self.groundTruthList = []
        with open(self.groundTruthListdir) as f:
            self.groundTruthList = f.read().splitlines()
        self.groundTruthList.sort()
        #print(self.groundTruthList)
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
            
            baseName = os.path.basename(InputImageList[i])
            baseName = baseName.split('.')[0]
            fn = result_dir + '/' + baseName + '_Label.nii.gz'
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
        
        # print(self.predictList)
        # print(self.groundTruthList)
        
        if len(self.predictList)!= len(self.groundTruthList):
            raise ValueError('the num of predicted images\
            should match that of the ground truth iamges')
        self.listLength = len(self.predictList)
        
        diceScore = np.zeros((self.listLength,))
        accScore = np.zeros((self.listLength,))
        precisionScore = np.zeros((self.listLength,))
        recallScore = np.zeros((self.listLength,))
        hausdorff_dis = np.zeros((self.listLength,))
        vol_diff = np.zeros((self.listLength,))
        for i in range(self.listLength):
            
            ImgroundTruth = sitk.ReadImage(self.groundTruthList[i])
            TmGT_array = sitk.GetArrayFromImage(ImgroundTruth)
            y_true = np.reshape(TmGT_array,-1)            

            ImPred = sitk.ReadImage(self.predictList[i])
            ImPred_array = sitk.GetArrayFromImage(ImPred)
            y_pred = np.reshape(ImPred_array,-1)
            
            # diceScore[i] = f1_score(y_true, y_pred)
            precisionScore[i], recallScore[i], diceScore[i], _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            
            accScore[i] = accuracy_score(y_true, y_pred)

            # true_bool = np.zeros(y_true.shape, dtype=bool)
            # pred_bool = np.zeros(y_pred.shape, dtype=bool)
            # true_bool[y_true > 0] = True
            # pred_bool[y_pred > 0] = True

            hausdorff_dis[i] = hausdorff_distance(TmGT_array.astype(bool), ImPred_array.astype(bool))
            vol_diff[i] = average_volume_difference(y_true, y_pred)

        dice_Statistics = {}
        dice_Statistics['mean'] = np.mean(diceScore)
        dice_Statistics['std'] = np.std(diceScore)
        dice_Statistics['max'] = np.amax(diceScore)
        dice_Statistics['min'] = np.amin(diceScore)

        precision_Statistics = {}
        precision_Statistics['mean'] = np.mean(precisionScore)
        precision_Statistics['std'] = np.std(precisionScore)
        precision_Statistics['max'] = np.amax(precisionScore)
        precision_Statistics['min'] = np.amin(precisionScore)

        recall_Statistics = {}
        recall_Statistics['mean'] = np.mean(recallScore)
        recall_Statistics['std'] = np.std(recallScore)
        recall_Statistics['max'] = np.amax(recallScore)
        recall_Statistics['min'] = np.amin(recallScore)

        acc_Statistics = {}
        acc_Statistics['mean'] = np.mean(accScore)
        acc_Statistics['std'] = np.std(accScore)
        acc_Statistics['max'] = np.amax(accScore)
        acc_Statistics['min'] = np.amin(accScore)

        hausdorff_Statistics = {}
        hausdorff_Statistics['mean'] = np.mean(hausdorff_dis)
        hausdorff_Statistics['std'] = np.std(hausdorff_dis)
        hausdorff_Statistics['max'] = np.amax(hausdorff_dis)
        hausdorff_Statistics['min'] = np.amin(hausdorff_dis)

        vol_diff_Statistics = {}
        vol_diff_Statistics['mean'] = np.mean(vol_diff)
        vol_diff_Statistics['std'] = np.std(vol_diff)
        vol_diff_Statistics['max'] = np.amax(vol_diff)
        vol_diff_Statistics['min'] = np.amin(vol_diff)

        print('Dice scores: ', diceScore)
        print('Dice score statistics: ', dice_Statistics)

        # print 'Accuracy scores: ', accScore
        # print 'Accuracy statistics: ', acc_Statistics

        print('Precision scores: ', precisionScore)
        print('Precision statistics: ', precision_Statistics)

        print('Recall scores: ', recallScore)
        print('Recall statistics: ', recall_Statistics)

        print('Hausdorff distances: ', hausdorff_dis)
        print('Hausdorff-distance statistics: ', hausdorff_Statistics)

        print('Volume Differences: ', vol_diff)
        print('Volume Difference statistics: ', vol_diff_Statistics)

        pass



