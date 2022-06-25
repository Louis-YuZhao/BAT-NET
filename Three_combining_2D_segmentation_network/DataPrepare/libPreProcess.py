#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
class for Brown Adipose Tissue preprocessing

Created on Fri Sep  8 08:26:03 2017

@author: louis
"""

import numpy as np
import SimpleITK as sitk
import os
import string
import subprocess
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt

#%%
def showImage(imageDir, bias, title):
    image = sitk.ReadImage(imageDir) # image in SITK format
    imArray = sitk.GetArrayFromImage(image) # get numpy array
    z_dim, x_dim, y_dim = imArray.shape # get 3D volume shape

    # display the image    
    fig = plt.figure(figsize=(15,5))    
    
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
    implot = plt.imshow(np.flipud(imArray[z_dim/2+bias_Z,:,:]), plt.cm.gray, origin = 'lower')
    plt.subplot(132)
    implot = plt.imshow(np.flipud(imArray[:,x_dim/2+bias_X,:]), plt.cm.spectral, origin = 'lower')
    plt.subplot(133)
    implot = plt.imshow(np.flipud(imArray[:,:,y_dim/2+bias_Y]), plt.cm.spectral, origin = 'lower')
  
    plt.title(title)
    
    del image, imArray
    
def showImageArray(imArray, bias, title):
    z_dim, x_dim, y_dim = imArray.shape # get 3D volume shape

    # display the image    
    fig = plt.figure(figsize=(15,5))
    
    bias_Z = bias['z']
    bias_X = bias['x']
    bias_Y = bias['y']
    
    if ((z_dim/2 + bias_Z) < 0) or ((z_dim/2 + bias_Z) > z_dim -1):
        raise ValueError('bias should range form' + str(-z_dim/2) + 'to' + str(z_dim-1-z_dim/2))
#    if ((z_dim/2 + bias_X) < 0) or ((z_dim/2 + bias_X) > z_dim -1):
#        raise ValueError('bias should range form' + str(-z_dim/2) + 'to' + str(z_dim-1-z_dim/2))
#    if ((z_dim/2 + bias_Y) < 0) or ((z_dim/2 + bias_Y) > z_dim -1):
#        raise ValueError('bias should range form' + str(-z_dim/2) + 'to' + str(z_dim-1-z_dim/2))
   
#    plt.subplot(131)
    implot = plt.imshow(np.flipud(imArray[z_dim/2+bias_Z,:,:]), plt.cm.gray, origin = 'lower')
#    plt.subplot(132)
#    implot = plt.imshow(np.flipud(imArray[:,x_dim/2+bias_X,:]), plt.cm.spectral, origin = 'lower')
#    plt.subplot(133)
#    implot = plt.imshow(np.flipud(imArray[:,:,y_dim/2+bias_Y]), plt.cm.spectral, origin = 'lower')
  
    plt.title(title)
    
    del imArray 
    
def largestConnectComponet_binary(inputArray):
    if (inputArray.dtype == np.bool) or (inputArray.dtype == np.uint8):
        raise ValueError('The datatype should be np.bool or np.uint8')
    blobs_labels = measure.label(inputArray, neighbors=4, background=0)
    unique, counts = np.unique(blobs_labels, return_counts=True)
    counts[unique == 0] = 0 # the count of the background should be zeros
    largestLabelItem = unique[np.argmax(counts)]
    idx = (blobs_labels == largestLabelItem)
    outputArray = np.zeros_like(inputArray).astype(inputArray.dtype)
    outputArray[idx] = np.uint8(1)
    return outputArray
    
#%%
class BATpreprocessingFF(object):
    """ class for Brown Adipose Tissue preprocessing """
    def __init__(self, outputdir, showTempImage):
        self.outputdir = outputdir
        self.showTempImage = showTempImage
        if not os.path.exists(self.outputdir):
            subprocess.call('mkdir ' + '-p ' + self.outputdir, shell=True)

    def inputImage(self, filepathFatFraction, filepathT2Star, filepathFat, filepathWater):
        
        # Fat fraction(FF)
        self.filepathFF = filepathFatFraction
        self.imageFF = sitk.ReadImage(self.filepathFF)
        self.arrayFF = sitk.GetArrayFromImage(self.imageFF)
        self.shapeFF = np.shape(self.arrayFF)
        print "maximum value of FatFraction: %f"%(np.amax(np.amax(np.amax(self.arrayFF))))
        print "minimum value of FatFraction: %f"%(np.amin(np.amin(np.amin(self.arrayFF))))
        
        # T2 star(T2S)
        self.filepathT2S = filepathT2Star
        self.imageT2S = sitk.ReadImage(self.filepathT2S)
        self.arrayT2S = sitk.GetArrayFromImage(self.imageT2S)
        self.shapeT2S = np.shape(self.arrayT2S)
        print "maximum value of T2Star: %f"%(np.amax(np.amax(np.amax(self.arrayT2S))))
        print "minimum value of T2Star: %f"%(np.amin(np.amin(np.amin(self.arrayT2S))))
        
        # Fat(F)
        self.filepathF = filepathFat
        self.imageF = sitk.ReadImage(self.filepathF)
        self.arrayF = sitk.GetArrayFromImage(self.imageF)
        self.shapeF = np.shape(self.arrayF)
        print "maximum value of Fat: %f"%(np.amax(np.amax(np.amax(self.arrayF))))
        print "minimum value of Fat: %f"%(np.amin(np.amin(np.amin(self.arrayF))))
        
        # Water(W)
        self.filepathW = filepathWater
        self.imageW = sitk.ReadImage(self.filepathW)
        self.arrayW = sitk.GetArrayFromImage(self.imageW)
        self.shapeW = np.shape(self.arrayW)
        print "maximum value of Water: %f"%(np.amax(np.amax(np.amax(self.arrayW))))
        print "minimum value of Water: %f"%(np.amin(np.amin(np.amin(self.arrayW))))        

    def removeDataByThreshold(self, threshold, ifLogOutput):
        """ canculate the FF map """
        self.arrayFF[self.arrayFF < threshold[0]] = 0
        self.arrayFF[self.arrayFF > threshold[1]] = 0

        self.imageFF = sitk.GetImageFromArray(self.arrayFF)
        self.imageFF.SetOrigin(self.imageFF.GetOrigin())                               
        self.imageFF.SetSpacing(self.imageFF.GetSpacing())                                
        self.imageFF.SetDirection(self.imageFF.GetDirection())   
                        
        if ifLogOutput != False:
            dirname = self.outputdir + '/'            
            name, ext = os.path.splitext(self.filepathFF)
            inBaseName = os.path.basename(name)
            outBaseName = string.join(inBaseName.split("_")[0:-1], "_")
            outputdirection = dirname + outBaseName + '_THR' + '.nrrd'             
            sitk.WriteImage(self.imageFF, outputdirection)
        else:
            return self.arrayFF
    
    def removeBackgroundmap(self, threshold, ifLogOutput):
        """ remove background of FF map 
            remove the bone and the air
        """
        self.waterAndfatArray = self.arrayW + self.arrayF
        idx = (self.waterAndfatArray <= threshold) # idx of the bone and the air area. 
        self.arrayFF[idx] = 0
        self.imageFF = sitk.GetImageFromArray(self.arrayFF)
        self.imageFF.SetOrigin(self.imageFF.GetOrigin())                               
        self.imageFF.SetSpacing(self.imageFF.GetSpacing())                                
        self.imageFF.SetDirection(self.imageFF.GetDirection())   

        if ifLogOutput != False:
            dirname = self.outputdir + '/'            
            name, ext = os.path.splitext(self.filepathFF)
            inBaseName = os.path.basename(name)
            outBaseName = string.join(inBaseName.split("_")[0:-1], "_")
            outputdirection = dirname + outBaseName + '_FF_RBG' + '.nrrd'             
            sitk.WriteImage(self.imageFF, outputdirection)
        else:
            return self.arrayFF
        
    def SaveWaterPlusFat(self, ifLogOutput=True):
        # restore the water plus fat array
        waterAndfatImage = sitk.GetImageFromArray(self.waterAndfatArray)
        waterAndfatImage.SetOrigin(self.imageFF.GetOrigin())                               
        waterAndfatImage.SetSpacing(self.imageFF.GetSpacing())                                
        waterAndfatImage.SetDirection(self.imageFF.GetDirection())   

        if ifLogOutput != False:
            dirname = self.outputdir + '/'            
            name, ext = os.path.splitext(self.filepathFF)
            inBaseName = os.path.basename(name)
            outBaseName = string.join(inBaseName.split("_")[0:-1], "_")
            outputdirection = dirname + outBaseName + '_waterANDfat' + '.nrrd'             
            sitk.WriteImage(waterAndfatImage, outputdirection)       
    
    def removeSkinVoxels(self, FFthreshold, T2Sthreshold, iternum, CCthreshold, filterSize,\
                         ClossingSize, ErosionSize, ifLogOutput):
        """ remove the skin voxel
        """
        # skin FFthreshold <= 30%, R2_star >= 80 s-1 (T2_star <= 0.0125S)
        idxFF = (self.arrayFF > 0) & (self.arrayFF <= FFthreshold)
        idxT2S = (self.arrayT2S > 0) & (self.arrayT2S <= T2Sthreshold)
        skinAreaIdx = (idxFF & idxT2S)
        unSkinAreaIdx = ~(idxFF & idxT2S)
        
        skinArea = np.zeros_like(self.arrayFF)
        skinArea[skinAreaIdx] = self.arrayFF[skinAreaIdx]
        unSkinArea = np.zeros_like(self.arrayFF)
        unSkinArea[unSkinAreaIdx] = self.arrayFF[unSkinAreaIdx]
        
        #------------------------------------------------------#
        # Showing the image for debuging
        bias={}
        bias['z']=-10
        bias['x']=0
        bias['y']=0
        # show the temp image
        if self.showTempImage:
            showImageArray(self.arrayFF, bias, 'FF')
            showImageArray(skinAreaIdx, bias, 'skinIdx')
            showImageArray(skinArea, bias, 'skinArea')
        
        firstErosionSkinArea = np.zeros_like(unSkinArea)
        secondClossingArea = np.zeros_like(unSkinArea)
        thirdErosionArea = np.zeros_like(unSkinArea)
        
        z,y,x = self.shapeFF
        
        ### Erosion
        for i in range(z):
            firstErosionSkinArea[i,:,:]= ndimage.binary_erosion(skinAreaIdx[i,:,:],\
            iterations = iternum).astype(unSkinAreaIdx.dtype)
        #------------------------------------------------------#
        # Showing the image for debuging
        if self.showTempImage:
            showImageArray(firstErosionSkinArea, bias, 'skinAreaIdx_Erosion')
        
        unSkinAreaIdxWithoutBackground = (unSkinArea > CCthreshold) 
        # CCthreshold used for expel the background part
        unSkinAreaIdxWithoutBackground[skinAreaIdx] = False
        unSkinAreaIdxAfterErosion = unSkinAreaIdxWithoutBackground + firstErosionSkinArea   
        #------------------------------------------------------#
        # Showing the image for debuging
        if self.showTempImage:
            showImageArray(unSkinAreaIdxAfterErosion, bias, 'unSkinAreaIdxAfterErosion') 
        
        ### largestConnectedComponent
        largestConnectedComponent = largestConnectComponet_binary(unSkinAreaIdxAfterErosion)        
        #------------------------------------------------------#
        # Showing the image for debuging
        if self.showTempImage:
            showImageArray(largestConnectedComponent, bias, 'largestConnectedComponent')
        
        ### closing and median filter
        for i in range(z):
            secondClossingArea[i,:,:] = ndimage.binary_closing(largestConnectedComponent[i,:,:],\
            structure = ClossingSize).astype(unSkinAreaIdx.dtype)
  
            secondClossingArea[i,:,:] = ndimage.filters.median_filter(secondClossingArea[i,:,:], size=filterSize,\
            footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
            
        #------------------------------------------------------#
        # Showing the image for debuging
        if self.showTempImage:
            showImageArray(secondClossingArea, bias, 'binary_closing')        
        ### filling holes
        for i in range(z):
            secondClossingArea[i,:,:] = ndimage.binary_fill_holes(secondClossingArea[i,:,:]).astype(unSkinAreaIdx.dtype)
            
        #------------------------------------------------------#
        # Showing the image for debuging
        if self.showTempImage:
            showImageArray(secondClossingArea, bias, 'fill_holes')
        
        ### erosion
        for i in range(z):
            thirdErosionArea[i,:,:] = ndimage.binary_erosion(secondClossingArea[i,:,:],\
            structure = ErosionSize).astype(unSkinAreaIdx.dtype)     
        
        #---------------------------#
        # Showing the image for debuging
        if self.showTempImage:
            showImageArray(thirdErosionArea, bias, 'binary_erosion')
        
        finalIndex = (thirdErosionArea > 0)
        self.arrayFF [~finalIndex] = 0        
       
        #------------------------------------------------------#
        # Showing the image for debuging
        if self.showTempImage:
            showImageArray(self.arrayFF, bias, 'finalImage')
        
        # restore the wholeBodyMask
        wholeBodyMask = sitk.GetImageFromArray(thirdErosionArea)
        wholeBodyMask.SetOrigin(self.imageFF.GetOrigin())                               
        wholeBodyMask.SetSpacing(self.imageFF.GetSpacing())                                
        wholeBodyMask.SetDirection(self.imageFF.GetDirection())   

        if ifLogOutput != False:
            dirname = self.outputdir + '/'            
            name, ext = os.path.splitext(self.filepathFF)
            inBaseName = os.path.basename(name)
            outBaseName = string.join(inBaseName.split("_")[0:-1], "_")
            outputdirection = dirname + outBaseName + '_wholeBodyMask' + '.nrrd'             
            sitk.WriteImage(wholeBodyMask, outputdirection)  
        
        self.imageFF = sitk.GetImageFromArray(self.arrayFF)
        self.imageFF.SetOrigin(self.imageFF.GetOrigin())                               
        self.imageFF.SetSpacing(self.imageFF.GetSpacing())                                
        self.imageFF.SetDirection(self.imageFF.GetDirection())   

        if ifLogOutput != False:
            dirname = self.outputdir + '/'            
            name, ext = os.path.splitext(self.filepathFF)
            inBaseName = os.path.basename(name)
            outBaseName = string.join(inBaseName.split("_")[0:-1], "_")
            outputdirection = dirname + outBaseName + '_FF_RSkin' + '.nrrd'             
            sitk.WriteImage(self.imageFF, outputdirection)
        else:
            return self.arrayFF


class BATpreprocessingT2S(object):
    """ class for Brown Adipose Tissue """
    def __init__(self, outputdir):
        self.outputdir = outputdir
        if not os.path.exists(self.outputdir):
            subprocess.call('mkdir ' + '-p ' + self.outputdir, shell=True)
       
    def inputImage(self, filepathFatFraction, filepathT2Star, filepathFat, filepathWater):
        
        # Fat fraction(FF)
        self.filepathFF = filepathFatFraction
        self.imageFF = sitk.ReadImage(self.filepathFF)
        self.arrayFF = sitk.GetArrayFromImage(self.imageFF)
        self.shapeFF = np.shape(self.arrayFF)
        print "maximum value of FatFraction: %f"%(np.amax(np.amax(np.amax(self.arrayFF))))
        print "minimum value of FatFraction: %f"%(np.amin(np.amin(np.amin(self.arrayFF))))
        
        # T2 star(T2S)
        self.filepathT2S = filepathT2Star
        self.imageT2S = sitk.ReadImage(self.filepathT2S)
        self.arrayT2S = sitk.GetArrayFromImage(self.imageT2S)
        self.shapeT2S = np.shape(self.arrayT2S)
        print "maximum value of T2Star: %f"%(np.amax(np.amax(np.amax(self.arrayT2S))))
        print "minimum value of T2Star: %f"%(np.amin(np.amin(np.amin(self.arrayT2S))))
        
        # Fat(F)
        self.filepathF = filepathFat
        self.imageF = sitk.ReadImage(self.filepathF)
        self.arrayF = sitk.GetArrayFromImage(self.imageF)
        self.shapeF = np.shape(self.arrayF)
        print "maximum value of Fat: %f"%(np.amax(np.amax(np.amax(self.arrayF))))
        print "minimum value of Fat: %f"%(np.amin(np.amin(np.amin(self.arrayF))))
        
        # Water(W)
        self.filepathW = filepathWater
        self.imageW = sitk.ReadImage(self.filepathW)
        self.arrayW = sitk.GetArrayFromImage(self.imageW)
        self.shapeW = np.shape(self.arrayW)
        print "maximum value of Water: %f"%(np.amax(np.amax(np.amax(self.arrayW))))
        print "minimum value of Water: %f"%(np.amin(np.amin(np.amin(self.arrayW))))    
    
    def removeBackgroundmap(self, threshold, ifLogOutput):
        """ remove background of WF map 
            remove the bone and the air
        """
        waterAndfatArray = self.arrayW + self.arrayF
        idx = (waterAndfatArray <= threshold) # idx of the bone and the air area. 
        self.arrayT2S = sitk.GetArrayFromImage(self.imageT2S)
        self.arrayT2S[idx] = 0

        self.imageT2S = sitk.GetImageFromArray(self.arrayT2S)
        self.imageT2S.SetOrigin(self.imageT2S.GetOrigin())                               
        self.imageT2S.SetSpacing(self.imageT2S.GetSpacing())                                
        self.imageT2S.SetDirection(self.imageT2S.GetDirection())   

        if ifLogOutput != False:
            dirname = self.outputdir + '/'             
            name, ext = os.path.splitext(self.filepathT2S)
            inBaseName = os.path.basename(name)
            outBaseName = string.join(inBaseName.split("_")[0:-2], "_")
            outputdirection = dirname + outBaseName + '_T2S_RBG' + '.nrrd'             
            sitk.WriteImage(self.imageT2S, outputdirection)
        else:
            return self.imageT2S
  
    def removeSkinVoxels(self, FFthreshold, T2Sthreshold, iternum, filterSize,\
                         ClossingSize, ErosionSize, ifLogOutput):
        """ remove the skin voxel
        """
        # skin FFthreshold <= 30%, R2_star >= 80 s-1 (T2_star <= 0.0125S)
        # skin FFthreshold <= 30%, R2_star >= 80 s-1 (T2_star <= 0.0125S)
        idxFF = (self.arrayFF > 0) & (self.arrayFF <= FFthreshold)
        idxT2S = (self.arrayT2S > 0) & (self.arrayT2S <= T2Sthreshold)
        skinAreaIdx = (idxFF & idxT2S)
        unSkinAreaIdx = ~(idxFF & idxT2S)
        
        skinArea = np.zeros_like(self.arrayFF)
        skinArea[skinAreaIdx] = self.arrayFF[skinAreaIdx]
        unSkinArea = np.zeros_like(self.arrayFF)
        unSkinArea[unSkinAreaIdx] = self.arrayFF[unSkinAreaIdx]

        firstErosionSkinArea = np.zeros_like(unSkinArea)
        secondClossingArea = np.zeros_like(unSkinArea)
        thirdErosionArea = np.zeros_like(unSkinArea)        
        z,y,x = self.shapeFF

        ### Erosion
        for i in range(z):
            firstErosionSkinArea[i,:,:]= ndimage.binary_erosion(skinAreaIdx[i,:,:],\
            iterations = iternum).astype(unSkinAreaIdx.dtype)
        
        unSkinAreaIdxWithoutBackground = (unSkinArea > CCthreshold) 
        # CCthreshold used for expel the background part
        unSkinAreaIdxWithoutBackground[skinAreaIdx] = False
        unSkinAreaIdxAfterErosion = unSkinAreaIdxWithoutBackground + firstErosionSkinArea          
       
        ### largestConnectedComponent
        largestConnectedComponent = largestConnectComponet_binary(unSkinAreaIdxAfterErosion)        
        
        ### closing and median filter
        for i in range(z):
            secondClossingArea[i,:,:] = ndimage.binary_closing(largestConnectedComponent[i,:,:],\
            structure = ClossingSize).astype(unSkinAreaIdx.dtype)
  
            secondClossingArea[i,:,:] = ndimage.filters.median_filter(secondClossingArea[i,:,:], size=filterSize,\
            footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
                 
        ### filling holes
        for i in range(z):
            secondClossingArea[i,:,:] = ndimage.binary_fill_holes(secondClossingArea[i,:,:]).astype(unSkinAreaIdx.dtype)
        
        ### erosion
        for i in range(z):
            thirdErosionArea[i,:,:] = ndimage.binary_erosion(secondClossingArea[i,:,:],\
            structure = ErosionSize).astype(unSkinAreaIdx.dtype)     
        
        finalIndex = (thirdErosionArea > 0)
        self.arrayT2S [~finalIndex] = 0        
        
        self.imageT2S = sitk.GetImageFromArray(self.arrayT2S)
        self.imageT2S.SetOrigin(self.imageT2S.GetOrigin())                               
        self.imageT2S.SetSpacing(self.imageT2S.GetSpacing())                                
        self.imageT2S.SetDirection(self.imageT2S.GetDirection())   

        if ifLogOutput != False:
            dirname = self.outputdir + '/'                
            name, ext = os.path.splitext(self.filepathT2S)
            inBaseName = os.path.basename(name)
            outBaseName = string.join(inBaseName.split("_")[0:-2], "_")
            outputdirection = dirname + outBaseName + '_T2S_RSkin' + '.nrrd'            
            sitk.WriteImage(self.imageT2S, outputdirection)
        else:
            return self.imageT2S   

    def reduceNoise(self, filterSize, ifLogOutput):
        """ reduce noise by using median-filter
        """        
        x,y,z = np.shape(self.arrayT2S)
        inputarray = self.arrayT2S
        # 2D median-filter
        # for i in range(x):
        #     self.arrayT2S[i,:,:] = ndimage.filters.median_filter(inputarray[i,:,:], size=filterSize,\
        #                           footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
        
        # 3D median-filter
        self.arrayT2S = ndimage.filters.median_filter(inputarray, size=filterSize, \
            footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
       
        # Gaussion filter
        # self.arrayT2S = ndimage.filters.gaussian_filter(inputarray, sigma = 1.0, order=0,\
        #                 output=None, mode='reflect', cval=0.0, truncate=4.0)
        self.imageT2S = sitk.GetImageFromArray(self.arrayT2S)
        self.imageT2S.SetOrigin(self.imageT2S.GetOrigin())                               
        self.imageT2S.SetSpacing(self.imageT2S.GetSpacing())                                
        self.imageT2S.SetDirection(self.imageT2S.GetDirection())   

        if ifLogOutput != False:
            dirname = self.outputdir + '/'             
            name, ext = os.path.splitext(self.filepathT2S)
            inBaseName = os.path.basename(name)
            outBaseName = string.join(inBaseName.split("_")[0:-2], "_")
            outputdirection = dirname + outBaseName + '_T2S_Mfilter' + '.nrrd'             
            sitk.WriteImage(self.imageT2S, outputdirection)
        else:
            return self.imageT2S   