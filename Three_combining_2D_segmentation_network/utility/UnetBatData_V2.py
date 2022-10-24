#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
class for preparing the UNet Dataset with multi-modality
Created on Wed Dec 13 14:22:52 2017
Updated on Sunday Oct 28 2018
@author: louis
"""
import os
import numpy as np
import SimpleITK as sitk
import string
import random
import subprocess
import matplotlib.pyplot as plt

#%%
def GetListFromFolder(filename, endswith = ".nrrd"):
    imageList = []
    for fileItem in os.listdir(filename):
        if fileItem.endswith(endswith):
            imageList.append(filename + fileItem)
    imageList.sort()
    return imageList

def GetImageID(ImageList):
    ImageIDList = []
    for item in ImageList:
        filename = os.path.basename(item)
        checkItem = string.join((filename).split("_")[0:2], "_")
        ImageIDList.append(checkItem)
    return ImageIDList

def readTxtIntoList(filename):
    flist = []
    with open(filename) as f:
        flist = f.read().splitlines()
    return flist

def WriteListtoFile(filelist, filename):
    with open(filename, 'w') as f:
        for i in filelist:
            f.write(i+'\n')
    return 1

def showImageArray(imArray, bias, title):
    z_dim, x_dim, y_dim = imArray.shape # get 3D volume shape

    # display the image    
    plt.figure(figsize=(15,5))
    
    bias_Z = bias['z']
    bias_X = bias['x']
    bias_Y = bias['y']
    
    if ((z_dim/2 + bias_Z) < 0) or ((z_dim/2 + bias_Z) > z_dim -1):
        raise ValueError('bias should range form' + str(-z_dim/2) + 'to' + str(z_dim-1-z_dim/2))
    if ((z_dim/2 + bias_X) < 0) or ((z_dim/2 + bias_X) > z_dim -1):
        raise ValueError('bias should range form' + str(-z_dim/2) + 'to' + str(z_dim-1-z_dim/2))
    if ((z_dim/2 + bias_Y) < 0) or ((z_dim/2 + bias_Y) > z_dim -1):
        raise ValueError('bias should range form' + str(-z_dim/2) + 'to' + str(z_dim-1-z_dim/2))
   
    plt.subplot(131)
    plt.imshow(np.flipud(imArray[z_dim/2+bias_Z,:,:]), plt.cm.gray, origin = 'lower')
    plt.subplot(132)
    plt.imshow(np.flipud(imArray[:,x_dim/2+bias_X,:]), plt.cm.gray, origin = 'lower')
    plt.subplot(133)
    plt.imshow(np.flipud(imArray[:,:,y_dim/2+bias_Y]), plt.cm.gray, origin = 'lower')  
    plt.title(title)    
    del imArray 

#%%
class MulitModilityDataSplit(object):
    """ split the multi-modility imgages into training and test parts """
    def __init__(self, outputDir):
        self.IsThereMask = False
        self.outputDir = outputDir

    def inputImage(self, filepathFatFraction, filepathT2Star, filepathFat, filepathWater):
        
        # Fat fraction(FF)
        self.filepathFF = filepathFatFraction
        self.listFF = GetListFromFolder(self.filepathFF)
        self.ImageIDs = GetImageID(self.listFF)
        self.testID = []
        self.trainingID = []
        
        # T2 star(T2S)
        self.filepathT2S = filepathT2Star
        self.listT2S = GetListFromFolder(self.filepathT2S)
        
        # Fat(F)
        self.filepathF = filepathFat
        self.listF = GetListFromFolder(self.filepathF)
        
        # Water(W)
        self.filepathW = filepathWater
        self.listW = GetListFromFolder(self.filepathW)

        if len(self.listFF) == len(self.listT2S) == len(self.listF) == len(self.listW):
            self.ImageNum = len(self.listFF)
        else:
            raise ValueError('the length of the files should be same')
        for i in xrange(len(self.ImageIDs)):
            checkItem = self.ImageIDs[i]
            if not ((checkItem in self.listFF[i]) and (checkItem in self.listT2S[i])\
                and (checkItem in self.listF[i]) and (checkItem in self.listW[i])):
                raise ValueError(str(i)+' th do not match each other')

    def inputMask(self, filePathMask):
        # Mask
        self.filepathMask = filePathMask
        self.listMask = GetListFromFolder(self.filepathMask)
        self.IsThereMask = True
        self.matchCheck(self.listMask)
    
    def inputLabel(self, filepathLabel):
        # Label
        self.filepathLabel = filepathLabel
        self.listLabel = GetListFromFolder(self.filepathLabel)
        self.matchCheck(self.listLabel)
    
    def matchCheck(self, listCheck):
        # check whether the different lists match each other.
        if len(self.ImageIDs) == len(listCheck):
            ImageNum = len(self.ImageIDs)
        else:
            raise ValueError('the length of the files should be same')
        
        for i in xrange(ImageNum):
            checkItem = self.ImageIDs[i]
            if not(checkItem in listCheck[i]):
                raise ValueError(str(i)+'th do not match each other')
    def splitData(self, testNum):
        currentList = range(len(self.ImageIDs))
        random.shuffle(currentList)
        testID = currentList[:testNum]
        trainingID = currentList[testNum:]
        
        # train part
        trainList_FF = []
        trainList_T2S = []
        trainList_F =[]
        trainList_W =[]
        trainList_Label = []
        trainList_Mask = []
        for i in trainingID:
            self.trainingID.append(self.ImageIDs[i])
            trainList_FF.append(self.listFF[i])
            trainList_T2S.append(self.listT2S[i])
            trainList_F.append(self.listF[i])
            trainList_W.append(self.listW[i])
            trainList_Label.append(self.listLabel[i])
            if self.IsThereMask == True:
                trainList_Mask.append(self.listMask[i])

        # train part
        train_path = {}
        train_path['FF'] = os.path.join(self.outputDir, 'TrainingData/FF.txt')
        train_path['T2S'] = os.path.join(self.outputDir, 'TrainingData/T2S.txt')
        train_path['F'] = os.path.join(self.outputDir, 'TrainingData/F.txt')
        train_path['W'] = os.path.join(self.outputDir, 'TrainingData/W.txt')
        train_path['Label'] = os.path.join(self.outputDir, 'TrainingData/Label.txt')
        train_path['Mask'] = os.path.join(self.outputDir, 'TrainingData/Mask.txt')
        result_dir_pre = os.path.join(self.outputDir, 'TrainingData')
        if not os.path.exists(result_dir_pre):
            subprocess.call('mkdir ' + '-p ' + result_dir_pre, shell=True)
       
        WriteListtoFile(trainList_FF, train_path['FF'])
        WriteListtoFile(trainList_T2S, train_path['T2S'])
        WriteListtoFile(trainList_F, train_path['F'])
        WriteListtoFile(trainList_W, train_path['W'])
        WriteListtoFile(trainList_Label, train_path['Label'])
        if self.IsThereMask == True:
            WriteListtoFile(trainList_Mask, train_path['Mask'])

        # test part
        testList_FF = []
        testList_T2S = []
        testList_F =[]
        testList_W =[]
        testList_Label = []
        testList_Mask = []
        for i in testID:
            self.testID.append(self.ImageIDs[i])
            testList_FF.append(self.listFF[i])
            testList_T2S.append(self.listT2S[i])
            testList_F.append(self.listF[i])
            testList_W.append(self.listW[i])
            testList_Label.append(self.listLabel[i])
            if self.IsThereMask == True:
                testList_Mask.append(self.listMask[i])

        # test part
        test_path = {}
        test_path['FF'] = os.path.join(self.outputDir, 'TestData/FF.txt')
        test_path['T2S'] = os.path.join(self.outputDir, 'TestData/T2S.txt')
        test_path['F'] = os.path.join(self.outputDir, 'TestData/F.txt')
        test_path['W'] = os.path.join(self.outputDir, 'TestData/W.txt')
        test_path['Label'] = os.path.join(self.outputDir, 'TestData/Label.txt')
        test_path['Mask'] = os.path.join(self.outputDir, 'TestData/Mask.txt')
        result_dir_pre = os.path.join(self.outputDir, 'TestData')
        if not os.path.exists(result_dir_pre):
            subprocess.call('mkdir ' + '-p ' + result_dir_pre, shell=True)        
        
        WriteListtoFile(testList_FF, test_path['FF'])
        WriteListtoFile(testList_T2S, test_path['T2S'])
        WriteListtoFile(testList_F, test_path['F'])
        WriteListtoFile(testList_W, test_path['W'])
        WriteListtoFile(testList_Label, test_path['Label'])
        if self.IsThereMask == True:
            WriteListtoFile(testList_Mask, test_path['Mask'])
        
        return train_path, test_path
#%%
class UnetBatDataPreprocessing(object):
    """ class for Brown Adipose Tissue preprocessing """
    def __init__(self):
        pass
    def inputImage(self, filepathFatFraction, filepathT2Star, filepathFat, filepathWater):
        
        # Water fraction(FF)
        self.filepathFF = filepathFatFraction
        self.listFF = readTxtIntoList(self.filepathFF)
        
        # T2 star(T2S)
        self.filepathT2S = filepathT2Star
        self.listT2S = readTxtIntoList(self.filepathT2S)
        
        # Fat(F)
        self.filepathF = filepathFat
        self.listF = readTxtIntoList(self.filepathF)
        
        # Water(W)
        self.filepathW = filepathWater
        self.listW = readTxtIntoList(self.filepathW)
        
        print('Num of the FF, T2S, F, W:'+str((len(self.listFF),len(self.listT2S),len(self.listF),len(self.listW))))

        if len(self.listFF) == len(self.listT2S) == len(self.listF) == len(self.listW):
            self.ImageNum = len(self.listFF)
        else:
            raise ValueError('the length of the files should be same')
        for i in xrange(self.ImageNum):
            checkItem = string.join((os.path.basename(self.listFF[i])).split("_")[1:2], "_")
            print('the checkItem is :', checkItem)
            if not ((checkItem in self.listFF[i]) and (checkItem in self.listT2S[i])\
                and (checkItem in self.listF[i]) and (checkItem in self.listW[i])):
                raise ValueError(str(i)+' th do not match each other')

    def inputMask(self, filePathMask):
        # Mask
        self.filepathMask = filePathMask
        self.listMask = readTxtIntoList(self.filepathMask)
        self.IsThereMask = True
    
    def inputLabel(self, filepathLabel):
        # Label
        self.filepathLabel = filepathLabel
        self.listLabel = readTxtIntoList(self.filepathLabel)
    
    def matchCheck(self, listOne, listTwo):
        # check whether the different lists match each other.
        if len(listOne) == len(listTwo):
            ImageNum = len(listOne)
        else:
            raise ValueError('the length of the files should be same')
        
        for i in xrange(ImageNum):
            checkItem = string.join((os.path.basename(listOne[i])).split("_")[1:2], "_")
            if not((checkItem in listOne[i]) and (checkItem in listTwo[i])):
                raise ValueError(str(i)+'th do not match each other')

    def ReadVolumeDataWithoutNorm(self, imageList, outputTitle, dataType): 
  
        total = len(imageList)
        
        print('Loading begin.')
        img = sitk.ReadImage(imageList[0])
        imgsArray = sitk.GetArrayFromImage(img)
        imgsArray = imgsArray[np.newaxis,:]

        for i in xrange (1,total):        
            img = sitk.ReadImage(imageList[i])
            tempArray = sitk.GetArrayFromImage(img)
            tempArray = tempArray[np.newaxis,:]
            imgsArray = np.concatenate((imgsArray, tempArray), axis=0)                
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1        
        imgsArray.astype(dataType)
        np.save(outputTitle, imgsArray)
        print('image array shape:' + str(np.shape(imgsArray)))
        print('Saving to .npy files done.')

    def ReadVolumeDataNormEachPerson(self, imageList, outputTitle, dataType):   

        total = len(imageList)

        print('Loading begin.')
        img = sitk.ReadImage(imageList[0])
        imgsArray = sitk.GetArrayFromImage(img)
        mean = np.mean(imgsArray)  # mean for data centering
        std = np.std(imgsArray)  # std for data normalization
        imgsArray -= mean
        imgsArray /= std
        imgsArray = imgsArray[np.newaxis,:]

        for i in xrange (1,total):        
            img = sitk.ReadImage(imageList[i])
            tempArray = sitk.GetArrayFromImage(img)
            mean = np.mean(tempArray)  # mean for data centering
            std = np.std(tempArray)  # std for data normalization
            tempArray -= mean
            tempArray /= std
            tempArray = tempArray[np.newaxis,:]
            imgsArray = np.concatenate((imgsArray, tempArray), axis=0)                
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1        
        imgsArray.astype(dataType)
        np.save(outputTitle, imgsArray)
        print('Saving to .npy files done.')

    def ReadVolumeDataNormEachPersonWithinMask(self, imageList, maskList, outputTitle, dataType):
        # collect the image file path
        total = len(imageList)

        # collect the mask file path
        if self.IsThereMask != True:
            raise ValueError('There should be mask data, please recheck this') 

        # check whether the image matches the mask
        for i in xrange(total):
            if (string.join((imageList[i]).split("_")[0:1], "_") != string.join((maskList[i]).split("_")[0:1], "_")) \
                or ('Rotated' in imageList[i]) != ('Rotated' in maskList[i]):
                raise ValueError('the'+ str(i) + 'th image and mask should match each other!')                
        
        print('Loading begin.')
        img = sitk.ReadImage(imageList[0])
        imgsArray = sitk.GetArrayFromImage(img)
        mask = sitk.ReadImage(maskList[0])
        maskArray = sitk.GetArrayFromImage(mask)
        temp = imgsArray[maskArray>0]
        mean = np.mean(temp)  # mean for data centering
        std = np.std(temp)  # std for data normalization
        imgsArray[maskArray>0] -= mean
        imgsArray[maskArray>0] /= std
        imgsArray[maskArray<0.8] = 0
        imgsArray = imgsArray[np.newaxis,:]

        for i in xrange (1,total):
            img = sitk.ReadImage(imageList[i])
            tempArray = sitk.GetArrayFromImage(img)
            mask = sitk.ReadImage(maskList[i])
            maskArray = sitk.GetArrayFromImage(mask)
            temp = tempArray[maskArray>0]
            mean = np.mean(temp)  # mean for data centering
            std = np.std(temp)  # std for data normalization
            tempArray[maskArray>0] -= mean
            tempArray[maskArray>0] /= std
            tempArray[maskArray<0.8] = 0
            tempArray = tempArray[np.newaxis,:]
            imgsArray = np.concatenate((imgsArray, tempArray), axis=0)                
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        
        imgsArray.astype(dataType)
        np.save(outputTitle, imgsArray)
        print('Saving to .npy files done.')
        
    def ReadVolumeDataNormEachPersonWithinMaskV2(self, imageList, maskList, outputTitle, dataType):
        # collect the image file path
        total = len(imageList)

        # collect the mask file path
        if self.IsThereMask != True:
            raise ValueError('There should be mask data, please recheck this') 

        # check wether the image matches the mask
        for i in xrange(total):
            if (string.join((imageList[i]).split("_")[0:1], "_") != string.join((maskList[i]).split("_")[0:1], "_")) \
                or ('Rotated' in imageList[i]) != ('Rotated' in maskList[i]):
                raise ValueError('the'+ str(i) + 'th image and mask should match each other!')                
        
        print('Loading begin.')
        img = sitk.ReadImage(imageList[0])
        imgsArray = sitk.GetArrayFromImage(img)
        mask = sitk.ReadImage(maskList[0])
        maskArray = sitk.GetArrayFromImage(mask)
        temp = imgsArray[maskArray>0]
        maxItem = np.max(temp)  
        minItem = np.min(temp)  
        imgsArray[maskArray>0] -= minItem
        imgsArray[maskArray>0] /= (maxItem - minItem)
        imgsArray[maskArray<0.8] = 0
        imgsArray = imgsArray[np.newaxis,:]

        for i in xrange (1,total):
            img = sitk.ReadImage(imageList[i])
            tempArray = sitk.GetArrayFromImage(img)
            mask = sitk.ReadImage(maskList[i])
            maskArray = sitk.GetArrayFromImage(mask)
            temp = tempArray[maskArray>0]
            maxItem = np.max(temp)  
            minItem = np.min(temp)  
            tempArray[maskArray>0] -= minItem
            tempArray[maskArray>0] /= (maxItem - minItem)
            tempArray[maskArray<0.8] = 0
            tempArray = tempArray[np.newaxis,:]
            imgsArray = np.concatenate((imgsArray, tempArray), axis=0)                
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        
        imgsArray.astype(dataType)
        np.save(outputTitle, imgsArray)
        print('Saving to .npy files done.')
        
    def ReadVolumeDataNormEachPersonWithinMaskV3(self, imageList, maskList, outputTitle, dataType):
        # collect the image file path
        total = len(imageList)

        # collect the mask file path
        if self.IsThereMask != True:
            raise ValueError('There should be mask data, please recheck this') 

        # check wether the image matches the mask
        for i in xrange(total):
            if (string.join((imageList[i]).split("_")[0:1], "_") != string.join((maskList[i]).split("_")[0:1], "_")) \
                or ('Rotated' in imageList[i]) != ('Rotated' in maskList[i]):
                raise ValueError('the'+ str(i) + 'th image and mask should match each other!')                
        
        print('Loading begin.')
        img = sitk.ReadImage(imageList[0])
        imgsArray = sitk.GetArrayFromImage(img)
        mask = sitk.ReadImage(maskList[0])
        maskArray = sitk.GetArrayFromImage(mask)
        temp = imgsArray[maskArray>0]
        maxItem = np.max(temp)  
        minItem = np.min(temp)  
        imgsArray[maskArray>0] -= minItem
        imgsArray[maskArray>0] /= (maxItem - minItem)
        imgsArray[maskArray<0.8] = 0
        
        temp = imgsArray[maskArray>0]
        mean = np.mean(temp)  # mean for data centering
        std = np.std(temp)  # std for data normalization
        imgsArray[maskArray>0] -= mean
        imgsArray[maskArray>0] /= std
        imgsArray[maskArray<0.8] = 0
        imgsArray = imgsArray[np.newaxis,:]

        for i in xrange (1,total):
            img = sitk.ReadImage(imageList[i])
            tempArray = sitk.GetArrayFromImage(img)
            mask = sitk.ReadImage(maskList[i])
            maskArray = sitk.GetArrayFromImage(mask)
            temp = tempArray[maskArray>0]
            maxItem = np.max(temp)  
            minItem = np.min(temp)  
            tempArray[maskArray>0] -= minItem
            tempArray[maskArray>0] /= (maxItem - minItem)
            tempArray[maskArray<0.8] = 0
            
            temp = tempArray[maskArray>0]
            mean = np.mean(temp)  # mean for data centering
            std = np.std(temp)  # std for data normalization
            tempArray[maskArray>0] -= mean
            tempArray[maskArray>0] /= std
            tempArray[maskArray<0.8] = 0
            tempArray = tempArray[np.newaxis,:]
            
            imgsArray = np.concatenate((imgsArray, tempArray), axis=0)                
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        
        imgsArray.astype(dataType)
        np.save(outputTitle, imgsArray)
        print('Saving to .npy files done.')
    
    def ReadLabelData(self, outputTitle, dataType, labelList):   
        total = len(labelList)

        img = sitk.ReadImage(labelList[0])
        imgsArray = sitk.GetArrayFromImage(img)
        imgsArray = imgsArray[np.newaxis,:]

        for i in xrange (1,total):        
            img = sitk.ReadImage(labelList[i])
            tempArray = sitk.GetArrayFromImage(img)
            tempArray = tempArray[np.newaxis,:]
            imgsArray = np.concatenate((imgsArray, tempArray), axis=0)                
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        print('Loading done.')
        imgsArray.astype(dataType)
        np.save(outputTitle, imgsArray)
        print('Saving to .npy files done.')
    
    def RecordImageID(self, imageList, outputTitle ='imgs_id_test.npy'):
        total = len(imageList)         
        img_id = imageList[0].split('/')[-1]
        imgs_id = [img_id.split('.')[0]]
        for i in xrange (1,total):
            img_id = imageList[i].split('/')[-1]
            img_id = img_id.split('.')[0]     
            imgs_id.append(img_id)
        np.save(outputTitle, imgs_id)    
    