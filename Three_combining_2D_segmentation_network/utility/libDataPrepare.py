#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:31:16 2018

@author: user
"""
from __future__ import print_function

import os
import numpy as np
import utility.UnetBatData_V2 as UBD

def DatasplitFile(data_path, testNum, outputDir):    
    Datasplit = UBD.MulitModilityDataSplit(outputDir)
    Datasplit.inputImage(data_path['FF'],data_path['T2S'],data_path['F'],data_path['W'])
    Datasplit.inputLabel(data_path['Label'])
    Datasplit.inputMask(data_path['Mask'])
    Datasplit.matchCheck(Datasplit.listLabel)
    Datasplit.matchCheck(Datasplit.listMask)
    train_path, test_path = Datasplit.splitData(testNum)
    
    return train_path, test_path

class dataPrepare(object):

    def __init__(self, config):

        self.img_z = config['dim_z']
        self.img_x = config['dim_x']
        self.img_y = config['dim_y']
        self.dataDim = config['dataDim']

        # normalization
        self.NormType = config['NormType']
        self.IfglobalNorm = config['IfglobalNorm']
        if self.NormType == 0:
            self.IsThereMask = False 
        else:
            self.IsThereMask = True

    def create_train_data(self, train_path,tempStore):
        '''
        NormType = 2，3，4 : NormWithinMask
        NormType = 1 : NormWholeBody
        NormType = 0 : no Norm     
        '''    
        print('-'*30)
        print('Creating training images...')
        print('-'*30)

        UnetTrain = UBD.UnetBatDataPreprocessing()
        UnetTrain.inputImage(train_path['FF'],train_path['T2S'],train_path['F'],train_path['W'])
        UnetTrain.inputLabel(train_path['Label'])
        if self.IsThereMask == True:
            UnetTrain.inputMask(train_path['Mask'])
        UnetTrain.matchCheck(UnetTrain.listFF, UnetTrain.listLabel)
        
        if self.NormType == 4:
            # trainging volume # within Mask
            outputTitle = os.path.join(tempStore,'imgs_train_FF_WM.npy')
            UnetTrain.ReadVolumeDataNormEachPersonWithinMaskV3(UnetTrain.listFF, UnetTrain.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_T2S_WM.npy')
            UnetTrain.ReadVolumeDataNormEachPersonWithinMaskV3(UnetTrain.listT2S, UnetTrain.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_F_WM.npy')
            UnetTrain.ReadVolumeDataNormEachPersonWithinMaskV3(UnetTrain.listF, UnetTrain.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_W_WM.npy')
            UnetTrain.ReadVolumeDataNormEachPersonWithinMaskV3(UnetTrain.listW, UnetTrain.listMask, outputTitle, np.float32)            
        elif self.NormType == 3:
            # trainging volume # within Mask
            outputTitle = os.path.join(tempStore,'imgs_train_FF_WM.npy')
            UnetTrain.ReadVolumeDataNormEachPersonWithinMaskV2(UnetTrain.listFF, UnetTrain.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_T2S_WM.npy')
            UnetTrain.ReadVolumeDataNormEachPersonWithinMaskV2(UnetTrain.listT2S, UnetTrain.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_F_WM.npy')
            UnetTrain.ReadVolumeDataNormEachPersonWithinMaskV2(UnetTrain.listF, UnetTrain.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_W_WM.npy')
            UnetTrain.ReadVolumeDataNormEachPersonWithinMaskV2(UnetTrain.listW, UnetTrain.listMask, outputTitle, np.float32)
        elif self.NormType == 2:
            # trainging volume  # within Mask
            outputTitle = os.path.join(tempStore,'imgs_train_FF_WM.npy')
            UnetTrain.ReadVolumeDataNormEachPersonWithinMask(UnetTrain.listFF, UnetTrain.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_T2S_WM.npy')
            UnetTrain.ReadVolumeDataNormEachPersonWithinMask(UnetTrain.listT2S, UnetTrain.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_F_WM.npy')
            UnetTrain.ReadVolumeDataNormEachPersonWithinMask(UnetTrain.listF, UnetTrain.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_W_WM.npy')
            UnetTrain.ReadVolumeDataNormEachPersonWithinMask(UnetTrain.listW, UnetTrain.listMask, outputTitle, np.float32)
        elif self.NormType == 1:
            # trainging volume # whole Body
            outputTitle = os.path.join(tempStore,'imgs_train_FF_WB.npy')
            UnetTrain.ReadVolumeDataNormEachPerson(UnetTrain.listFF, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_T2S_WB.npy')
            UnetTrain.ReadVolumeDataNormEachPerson(UnetTrain.listT2S, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_F_WB.npy')
            UnetTrain.ReadVolumeDataNormEachPerson(UnetTrain.listF, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_W_WB.npy')
            UnetTrain.ReadVolumeDataNormEachPerson(UnetTrain.listW, outputTitle, np.float32)
        elif self.NormType == 0:
            # training volume without normlization
            outputTitle = os.path.join(tempStore,'imgs_train_FF.npy')
            UnetTrain.ReadVolumeDataWithoutNorm(UnetTrain.listFF, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_T2S.npy')
            UnetTrain.ReadVolumeDataWithoutNorm(UnetTrain.listT2S, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_F.npy')
            UnetTrain.ReadVolumeDataWithoutNorm(UnetTrain.listF, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_train_W.npy')
            UnetTrain.ReadVolumeDataWithoutNorm(UnetTrain.listW, outputTitle, np.float32)
        else:
            raise ValueError('NormType should by 0,1,2,3,4')

        # trainging label
        outputTitle = os.path.join(tempStore,'labs_train.npy')
        UnetTrain.ReadLabelData(outputTitle, np.uint32, UnetTrain.listLabel)
    
    def create_test_data(self, test_path, tempStore):
        
        print('-'*30)
        print('Creating test images...')
        print('-'*30)

        UnetTest = UBD.UnetBatDataPreprocessing()
        UnetTest.inputImage(test_path['FF'],test_path['T2S'],test_path['F'],test_path['W'])
        if self.IsThereMask == True:
            UnetTest.inputMask(test_path['Mask'])    
        outputTitle = os.path.join(tempStore,'imgs_id_test.npy')
        UnetTest.RecordImageID(UnetTest.listFF, outputTitle)
        if self.NormType == 4:
            # test volume # within Mask
            outputTitle = os.path.join(tempStore,'imgs_test_FF_WM.npy')
            UnetTest.ReadVolumeDataNormEachPersonWithinMaskV3(UnetTest.listFF, UnetTest.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_T2S_WM.npy')
            UnetTest.ReadVolumeDataNormEachPersonWithinMaskV3(UnetTest.listT2S, UnetTest.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_F_WM.npy')
            UnetTest.ReadVolumeDataNormEachPersonWithinMaskV3(UnetTest.listF, UnetTest.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_W_WM.npy')
            UnetTest.ReadVolumeDataNormEachPersonWithinMaskV3(UnetTest.listW, UnetTest.listMask, outputTitle, np.float32)
        elif self.NormType == 3:
            # test volume # within Mask
            outputTitle = os.path.join(tempStore,'imgs_test_FF_WM.npy')
            UnetTest.ReadVolumeDataNormEachPersonWithinMaskV2(UnetTest.listFF, UnetTest.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_T2S_WM.npy')
            UnetTest.ReadVolumeDataNormEachPersonWithinMaskV2(UnetTest.listT2S, UnetTest.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_F_WM.npy')
            UnetTest.ReadVolumeDataNormEachPersonWithinMaskV2(UnetTest.listF, UnetTest.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_W_WM.npy')
            UnetTest.ReadVolumeDataNormEachPersonWithinMaskV2(UnetTest.listW, UnetTest.listMask, outputTitle, np.float32)
        elif self.NormType == 2:
            # test volume # within MaskwholeBodyMask
            outputTitle = os.path.join(tempStore,'imgs_test_FF_WM.npy')
            UnetTest.ReadVolumeDataNormEachPersonWithinMask(UnetTest.listFF, UnetTest.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_T2S_WM.npy')
            UnetTest.ReadVolumeDataNormEachPersonWithinMask(UnetTest.listT2S, UnetTest.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_F_WM.npy')
            UnetTest.ReadVolumeDataNormEachPersonWithinMask(UnetTest.listF, UnetTest.listMask, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_W_WM.npy')
            UnetTest.ReadVolumeDataNormEachPersonWithinMask(UnetTest.listW, UnetTest.listMask, outputTitle, np.float32)
        elif self.NormType == 1:
            # test volume # whole Body
            outputTitle = os.path.join(tempStore,'imgs_test_FF_WB.npy')
            UnetTest.ReadVolumeDataNormEachPerson(UnetTest.listFF, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_T2S_WB.npy')
            UnetTest.ReadVolumeDataNormEachPerson(UnetTest.listT2S, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_F_WB.npy')
            UnetTest.ReadVolumeDataNormEachPerson(UnetTest.listF, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_W_WB.npy')
            UnetTest.ReadVolumeDataNormEachPerson(UnetTest.listW, outputTitle, np.float32)
        elif self.NormType == 0:
            # test volume # whole Body
            outputTitle = os.path.join(tempStore,'imgs_test_FF.npy')
            UnetTest.ReadVolumeDataWithoutNorm(UnetTest.listFF, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_T2S.npy')
            UnetTest.ReadVolumeDataWithoutNorm(UnetTest.listT2S, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_F.npy')
            UnetTest.ReadVolumeDataWithoutNorm(UnetTest.listF, outputTitle, np.float32)
            outputTitle = os.path.join(tempStore,'imgs_test_W.npy')
            UnetTest.ReadVolumeDataWithoutNorm(UnetTest.listW, outputTitle, np.float32)
        else:
            raise ValueError('NormType should be 0,1,2,3,4')

    def load_train_data(self, tempStore):
        imgs_train = {}
        if self.NormType == 4 or self.NormType == 3 or self.NormType == 2:
            # test volume  # within Mask
            imgs_train['FF'] = np.load(os.path.join(tempStore,'imgs_train_FF_WM.npy'))
            imgs_train['T2S'] = np.load(os.path.join(tempStore,'imgs_train_T2S_WM.npy'))
            imgs_train['F'] = np.load(os.path.join(tempStore,'imgs_train_F_WM.npy'))
            imgs_train['W'] = np.load(os.path.join(tempStore,'imgs_train_W_WM.npy'))
        elif self.NormType == 1:
            # test volume  # whole Body
            imgs_train['FF'] = np.load(os.path.join(tempStore,'imgs_train_FF_WB.npy'))
            imgs_train['T2S'] = np.load(os.path.join(tempStore,'imgs_train_T2S_WB.npy'))
            imgs_train['F'] = np.load(os.path.join(tempStore,'imgs_train_F_WB.npy'))
            imgs_train['W'] = np.load(os.path.join(tempStore,'imgs_train_W_WB.npy'))
        elif self.NormType == 0:
            # without Normlization
            imgs_train['FF'] = np.load(os.path.join(tempStore,'imgs_train_FF.npy'))
            imgs_train['T2S'] = np.load(os.path.join(tempStore,'imgs_train_T2S.npy'))
            imgs_train['F'] = np.load(os.path.join(tempStore,'imgs_train_F.npy'))
            imgs_train['W'] = np.load(os.path.join(tempStore,'imgs_train_W.npy'))
        else:
            raise ValueError('NormType should be 0,1,2,3,4')            
        imgs_label_train = np.load(os.path.join(tempStore,'labs_train.npy'))
        return imgs_train, imgs_label_train

    def del_train_data(self, tempStore):
        if self.NormType == 4 or self.NormType == 3 or self.NormType == 2:
            # test volume  # within Mask
            os.remove(os.path.join(tempStore,'imgs_train_FF_WM.npy'))
            os.remove(os.path.join(tempStore,'imgs_train_T2S_WM.npy'))
            os.remove(os.path.join(tempStore,'imgs_train_F_WM.npy'))
            os.remove(os.path.join(tempStore,'imgs_train_W_WM.npy'))
        elif self.NormType == 1:
            # test volume  # whole Body
            os.remove(os.path.join(tempStore,'imgs_train_FF_WB.npy'))
            os.remove(os.path.join(tempStore,'imgs_train_T2S_WB.npy'))
            os.remove(os.path.join(tempStore,'imgs_train_F_WB.npy'))
            os.remove(os.path.join(tempStore,'imgs_train_W_WB.npy'))
        elif self.NormType == 0:
            # without Normlization
            os.remove(os.path.join(tempStore,'imgs_train_FF.npy'))
            os.remove(os.path.join(tempStore,'imgs_train_T2S.npy'))
            os.remove(os.path.join(tempStore,'imgs_train_F.npy'))
            os.remove(os.path.join(tempStore,'imgs_train_W.npy'))
        else:
            raise ValueError('NormType should be 0,1,2,3,4')            
        os.remove(os.path.join(tempStore,'labs_train.npy'))
    
    def load_test_data(self, tempStore):    
        imgs_test = {}
        if self.NormType == 4 or self.NormType == 3 or self.NormType == 2:
            # test volume  # within Mask
            imgs_test['FF'] = np.load(os.path.join(tempStore,'imgs_test_FF_WM.npy'))
            imgs_test['T2S'] = np.load(os.path.join(tempStore,'imgs_test_T2S_WM.npy'))
            imgs_test['F'] = np.load(os.path.join(tempStore,'imgs_test_F_WM.npy'))
            imgs_test['W'] = np.load(os.path.join(tempStore,'imgs_test_W_WM.npy'))
        elif self.NormType == 1:
            # test volume  # whole Body
            imgs_test['FF'] = np.load(os.path.join(tempStore,'imgs_test_FF_WB.npy'))
            imgs_test['T2S'] = np.load(os.path.join(tempStore,'imgs_test_T2S_WB.npy'))
            imgs_test['F'] = np.load(os.path.join(tempStore,'imgs_test_F_WB.npy'))
            imgs_test['W'] = np.load(os.path.join(tempStore,'imgs_test_W_WB.npy'))
        elif self.NormType == 0:
            # without Normlization
            imgs_test['FF'] = np.load(os.path.join(tempStore,'imgs_test_FF.npy'))
            imgs_test['T2S'] = np.load(os.path.join(tempStore,'imgs_test_T2S.npy'))
            imgs_test['F'] = np.load(os.path.join(tempStore,'imgs_test_F.npy'))
            imgs_test['W'] = np.load(os.path.join(tempStore,'imgs_test_W.npy')) 

        imgs_id = np.load(os.path.join(tempStore,'imgs_id_test.npy'))
        return imgs_test, imgs_id

    def del_test_data(self, tempStore):    
        if self.NormType == 4 or self.NormType == 3 or self.NormType == 2:
            # test volume  # within Mask
            os.remove(os.path.join(tempStore,'imgs_test_FF_WM.npy'))
            os.remove(os.path.join(tempStore,'imgs_test_T2S_WM.npy'))
            os.remove(os.path.join(tempStore,'imgs_test_F_WM.npy'))
            os.remove(os.path.join(tempStore,'imgs_test_W_WM.npy'))
        elif self.NormType == 1:
            # test volume  # whole Body
            os.remove(os.path.join(tempStore,'imgs_test_FF_WB.npy'))
            os.remove(os.path.join(tempStore,'imgs_test_T2S_WB.npy'))
            os.remove(os.path.join(tempStore,'imgs_test_F_WB.npy'))
            os.remove(os.path.join(tempStore,'imgs_test_W_WB.npy'))
        elif self.NormType == 0:
            # without Normlization
            os.remove(os.path.join(tempStore,'imgs_test_FF.npy'))
            os.remove(os.path.join(tempStore,'imgs_test_T2S.npy'))
            os.remove(os.path.join(tempStore,'imgs_test_F.npy'))
            os.remove(os.path.join(tempStore,'imgs_test_W.npy')) 

    def preprocessDimZ(self, imgs):
        ChannelNum = len(imgs)
        # N img_Z, img_X, img_Y, Channel Number
        Num = imgs['FF'].shape[0]
        imgs_p = np.ndarray(((Num * self.img_z), self.img_x, self.img_y, ChannelNum), dtype=np.float32)
        channel = 0
        for key in imgs:
            for i in range(Num):
                for j in range(self.img_z):
                    imgs_p[((i*self.img_z)+j),:,:,channel] = imgs[key][i,j,:,:]
            channel = channel+1
        return imgs_p

    def preprocessDimX(self, imgs):
        ChannelNum = len(imgs)
        # N img_Z, img_X, img_Y, Channel Number
        Num = imgs['FF'].shape[0]
        imgs_p = np.ndarray(((Num * self.img_x), self.img_z, self.img_y, ChannelNum), dtype=np.float32)
        channel = 0
        for key in imgs:
            for i in range(Num):
                for j in range(self.img_x):
                    imgs_p[((i*self.img_x)+j),:,:,channel] = imgs[key][i,:,j,:]
            channel = channel+1
        return imgs_p

    def preprocessDimY(self, imgs):
        ChannelNum = len(imgs)
        # N img_Z, img_X, img_Y, Channel Number
        Num = imgs['FF'].shape[0]
        imgs_p = np.ndarray(((Num * self.img_y), self.img_z, self.img_x, ChannelNum), dtype=np.float32)
        channel = 0
        for key in imgs:
            for i in range(Num):
                for j in range(self.img_y):
                    imgs_p[((i*self.img_y)+j),:,:,channel] = imgs[key][i,:,:,j]
            channel = channel+1
        return imgs_p

    def preprocessLabel(self, imgs):
        Num = imgs.shape[0]

        if self.dataDim == 'z':

            imgs_p = np.ndarray(((Num * self.img_z), self.img_x, self.img_y, 1), dtype=np.float32)           
            for i in range(Num):
                for j in range(self.img_z):
                    imgs_p[((i*self.img_z)+j),:,:,0] = imgs[i,j,:,:]       
            return imgs_p
        
        elif self.dataDim == 'x':

            imgs_p = np.ndarray(((Num * self.img_x), self.img_z, self.img_y, 1), dtype=np.float32)
            for i in range(Num):
                for j in range(self.img_x):
                    imgs_p[((i*self.img_x)+j),:,:,0] = imgs[i,:,j,:]
            return imgs_p

        elif self.dataDim == 'y':

            imgs_p = np.ndarray(((Num * self.img_y), self.img_z, self.img_x, 1), dtype=np.float32)
            for i in range(Num):
                for j in range(self.img_y):
                    imgs_p[((i*self.img_y)+j),:,:,0] = imgs[i,:,:,j]
            return imgs_p
        else:
            raise ValueError('dataDim should be z, x or y')

    def trainPrepare(self, tempStore):
        imgs_train_dic, imgs_label_train = self.load_train_data(tempStore)
        if self.dataDim == 'z':
            imgs_train = self.preprocessDimZ(imgs_train_dic)
        elif self.dataDim == 'x':
            imgs_train = self.preprocessDimX(imgs_train_dic)
        elif self.dataDim == 'y':
            imgs_train = self.preprocessDimY(imgs_train_dic)
        else:
            raise ValueError('dataDim should be x, y, z')

        imgs_label_train = self.preprocessLabel(imgs_label_train)

        imgs_train = imgs_train.astype('float32')     
        imgs_label_train = imgs_label_train.astype(np.uint8)

        if self.IfglobalNorm == True:
            channelNum = imgs_train.shape[3]
            mean = np.zeros(channelNum,)
            std = np.zeros(channelNum,)
            for i in range(channelNum):
                mean[i] = np.mean(imgs_train[:,:,:,i])  # mean for data centering
                std[i] = np.std(imgs_train[:,:,:,i])  # std for data normalization
                imgs_train[:,:,:,i] -= mean[i]
                imgs_train[:,:,:,i] /= std[i]
            return imgs_train, imgs_label_train, mean, std
        
        return imgs_train, imgs_label_train
        
    def testPrepare(self, tempStore):
        #def testPrepare(self, tempStore, mean=None, std=None):
        imgs_test_dir, _ = self.load_test_data(tempStore)
        if self.dataDim == 'z':
            imgs_test = self.preprocessDimZ(imgs_test_dir)
        elif self.dataDim == 'x':
            imgs_test = self.preprocessDimX(imgs_test_dir)
        elif self.dataDim == 'y':
            imgs_test = self.preprocessDimY(imgs_test_dir)
        else:
            raise ValueError('dataDim should be x, y, z')
        imgs_test = imgs_test.astype('float32')
        
        if self.IfglobalNorm == True:
            channelNum = imgs_test.shape[3]
            mean = np.zeros(channelNum,)
            std = np.zeros(channelNum,)
            for i in range(channelNum):
                mean[i] = np.mean(imgs_test[:,:,:,i])  # mean for data centering
                std[i] = np.std(imgs_test[:,:,:,i])  # std for data normalization
                imgs_test[:,:,:,i] -= mean[i]
                imgs_test[:,:,:,i] /= std[i]
        return imgs_test
    def MultiChannelDataPrepare(self, train_path, test_path, tempStore):

        # train part
        self.create_train_data(train_path,tempStore)
        
        # test part     
        self.create_test_data(test_path,tempStore)
    
    def Train_and_predict_dataPrepare(self, tempStore):
        
        print('-'*30)
        print('Loading and preprocessing train data...')
        print('-'*30) 
        
        if self.IfglobalNorm == True:
            imgs_train, imgs_label_train, mean, std = self.trainPrepare(tempStore)
        elif self.IfglobalNorm == False:
            imgs_train, imgs_label_train = self.trainPrepare(tempStore)
            
        np.save(os.path.join(tempStore,'imgs_volume_train.npy'),imgs_train)
        np.save(os.path.join(tempStore,'imgs_label_train.npy'),imgs_label_train)
        
        print('-'*30)
        print('Loading and preprocessing test data...')
        print('-'*30)
        
        if self.IfglobalNorm == True:
            imgs_test = self.testPrepare(tempStore)
        elif self.IfglobalNorm == False:
            imgs_test = self.testPrepare(tempStore)    
        np.save(os.path.join(tempStore,'imgs_volume_test.npy'),imgs_test) 