#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
class for Brown Adipose Tissue 

Created on Fri Sep  8 08:26:03 2017

@author: louis
"""

import numpy as np
import os
import string
import subprocess
import SimpleITK as sitk
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects

#%%
def showImageArray(imArray, title):
    z_dim, x_dim, y_dim = imArray.shape # get 3D volume shape

    # display the image    
    fig = plt.figure(figsize=(15,5))    
   
    plt.subplot(131)
    implot = plt.imshow(np.flipud(imArray[z_dim/2,:,:]),plt.cm.gray)
    plt.subplot(132)
    implot = plt.imshow(np.flipud(imArray[:,x_dim/2,:]),plt.cm.spectral)
    plt.subplot(133)
    implot = plt.imshow(np.flipud(imArray[:,:,y_dim/2]),plt.cm.spectral)
  
    plt.title(title)
    
    del imArray
    
#%%    
class postRefinement(object):
    """ class for fine adjustment and refinement of segmentation according Dr.Dominik """
    def __init__(self, outputdir, IFshowImage=False):
        self.outputdir = outputdir
        self.IFshowImage = IFshowImage
        if not os.path.exists(self.outputdir):
            subprocess.call('mkdir ' + '-p ' + self.outputdir, shell=True)
    
    def inputImage(self, filepathLabel, filepathFatFraction, filepathT2Star, filepathFat, filepathWater):
        
        # label
        self.filepathLab = filepathLabel
        self.imageLab = sitk.ReadImage(self.filepathLab)
        self.origin = self.imageLab.GetOrigin()
        self.direction = self.imageLab.GetDirection()
        self.spacing = self.imageLab.GetSpacing()
        self.arrayLab = sitk.GetArrayFromImage(self.imageLab).astype(np.uint8)
        self.shapeLab = np.shape(self.arrayLab)
        
        #-----------------------------------------------------#
        # show the image for debuging the code
        if self.IFshowImage != False:
            showImageArray(self.arrayLab, 'label before refinement')

        # fat fraction(FF)
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

    def fineAdjustSegmentation(self, FFthresholdLow, FFthresholdHigh, T2Sthreshold, ifLogOutput):

        # Automated fine adjustment of segmentation
        # (FF<=50)=0, (FF>=110)=0, (T2_star<=10)=0 
        idxFFLow = (self.arrayFF <= FFthresholdLow)
        self.arrayLab[idxFFLow]= np.uint8(0)
        
        idxFFHigh = (self.arrayFF >= FFthresholdHigh)
        self.arrayLab[idxFFHigh]= np.uint8(0)
        
        idxR2S = (self.arrayT2S <= T2Sthreshold)
        self.arrayLab[idxR2S]= np.uint8(0)
        
        #-----------------------------------------------------#
        # show the image for debuging the code
        if self.IFshowImage != False:
            showImageArray(self.arrayLab, 'label after refinement')
             
        self.imageLab = sitk.GetImageFromArray(self.arrayLab)
        self.imageLab.SetOrigin(self.origin)                               
        self.imageLab.SetSpacing(self.spacing)                                
        self.imageLab.SetDirection(self.direction)
        
        if ifLogOutput != False:
            dirname = self.outputdir + '/'             
            name, ext = os.path.splitext(self.filepathLab)
            inBaseName = os.path.basename(name)
            outBaseName = string.join(inBaseName.split("_")[0:-1], "_")
            outputdirection = dirname + outBaseName + '_Lab_fineAdjust' + ext             
            sitk.WriteImage(self.imageLab, outputdirection)
            return outputdirection
        else:
            return self.arrayLab

    def removeBoneadnAir(self, threshold, ifLogOutput):
        """ remove the bone and the air of Lab map 
        """
        waterAndfatArray = self.arrayW + self.arrayF
        idx = (waterAndfatArray <= threshold) # idx of the bone and the air area. 
        self.arrayLab[idx] = np.uint8(0)
        
        #-----------------------------------------------------#
        # show the image for debuging the code
        if self.IFshowImage != False:
            showImageArray(self.arrayLab, 'label after removeBoneadnAir')
 
        self.imageLab = sitk.GetImageFromArray(self.arrayLab)
        self.imageLab.SetOrigin(self.origin)                               
        self.imageLab.SetSpacing(self.spacing)                                
        self.imageLab.SetDirection(self.direction)

        if ifLogOutput != False:
            dirname = self.outputdir + '/'             
            name, ext = os.path.splitext(self.filepathLab)
            inBaseName = os.path.basename(name)
            outBaseName = string.join(inBaseName.split("_")[0:-1], "_")
            outputdirection = dirname + outBaseName + '_Lab_RBandA' + ext             
            sitk.WriteImage(self.imageLab, outputdirection)
            return outputdirection
        else:
            return self.arrayLab       

    def finalRefine(self, dilationPara, ClossingPara, ErosionPara, removeSmallObj,ifLogOutput):
        """ remove the skin voxel
        """
        # Filling the holes of the binary image
        self.arrayLab = binary_fill_holes (self.arrayLab, structure=None,\
                        output=None, origin=0).astype(self.arrayLab.dtype)
        
#        #-----------------------------------------------------#
#        # show the image for debuging the code
        if self.IFshowImage != False:
            showImageArray(self.arrayLab, 'label after holeFilling') 
        
        if dilationPara['IfDilation'] == True:
            DilationSize = dilationPara['structure']
            iternum = dilationPara['iterations']
            
            self.arrayLab = ndimage.binary_dilation(self.arrayLab,\
            structure = DilationSize, iterations = iternum).astype(self.arrayLab.dtype)
        
        if ClossingPara['IfClossing'] == True:
            ClossingSize = ClossingPara['structure']
            iternum = ClossingPara['iterations']

            self.arrayLab = ndimage.binary_closing(self.arrayLab,\
            structure = ClossingSize, iterations = iternum).astype(self.arrayLab.dtype)

        if ErosionPara['IfErosion'] == True:
            ErosionSize = ErosionPara['structure']
            iternum = ErosionPara['iterations']

            self.arrayLab = ndimage.binary_erosion(self.arrayLab,\
            structure = ErosionSize, iterations = iternum).astype(self.arrayLab.dtype)
         
        min_size = removeSmallObj['min_size']
        connectivity = removeSmallObj['connectivity']    
        # Remove connected components smaller than the specified size.
        self.arrayLab = remove_small_objects(self.arrayLab, min_size, connectivity,\
                                              in_place=False).astype(self.arrayLab.dtype) 
#        #-----------------------------------------------------#
#        # show the image for debuging the code
        if self.IFshowImage != False:
            showImageArray(self.arrayLab, 'label after remove_small_objects')

        self.imageLab = sitk.GetImageFromArray(self.arrayLab)
        self.imageLab.SetOrigin(self.origin)                               
        self.imageLab.SetSpacing(self.spacing)                                
        self.imageLab.SetDirection(self.direction)

        if ifLogOutput != False:
            dirname = self.outputdir + '/'             
            name, ext = os.path.splitext(self.filepathLab)
            inBaseName = os.path.basename(name)
            outBaseName = string.join(inBaseName.split("_")[0:-1], "_")
            outputdirection = dirname + outBaseName + '_Lab_finalRefine' + ext             
            sitk.WriteImage(self.imageLab, outputdirection)
            return outputdirection
        else:
            return self.arrayLab       