#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
change .dcm data to .nrrd data

Created on Fri Sep  8 08:26:03 2017

@author: louis
"""

#%%
import os
import numpy as np
import SimpleITK as sitk

#%%
def readTxtIntoList(filename):
   flist = []
   with open(filename) as f:
         flist = f.read().splitlines()
   return flist

def Origin_computing(InputImageList):
    N = len(InputImageList)
    Origins = np.zeros((3,N))
    for i in range(N):
        image = sitk.ReadImage(InputImageList[i])
        Origins[:,i] = image.GetOrigin()
    return tuple(np.mean(Origins, axis=1))

def ReadFoldandSort(data_path):
    imageList = []
    for fileItem in os.listdir(data_path):
        if fileItem.endswith(".dcm"):
            imageList.append(os.path.join(data_path, fileItem))
    imageList.sort()
    return imageList

#%%
inputroot = '/media/data/louis/ResearchData/BrownAdiposeTissue/'

dirname = os.path.join(inputroot, 'ImageSplitDCM', 'ImageSplitF')
datalist = ReadFoldandSort(dirname)
# counting the origin
origin = Origin_computing(datalist)
OriginSet = origin
DirectiongSet = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
SpacingSet = (1, 1, 1)

for dir_i in datalist:
    im = sitk.ReadImage(dir_i)
    imArray = sitk.GetArrayFromImage(im)
    image = sitk.GetImageFromArray(imArray)
    image.SetOrigin(OriginSet)
    image.SetDirection(DirectiongSet)
    image.SetSpacing(SpacingSet)
    name, ext = os.path.splitext(dir_i)
    inBaseName = os.path.basename(name)
    outputdirection = os.path.join(dirname, inBaseName + '.nrrd')             
    sitk.WriteImage(image, outputdirection)
    
dirname = os.path.join(inputroot + 'ImageSplitW')
datalist = ReadFoldandSort(dirname)
for dir_i in datalist:
    im = sitk.ReadImage(dir_i)
    imArray = sitk.GetArrayFromImage(im)
    image = sitk.GetImageFromArray(imArray)
    image.SetOrigin(OriginSet)
    image.SetDirection(DirectiongSet)
    image.SetSpacing(SpacingSet)
    name, ext = os.path.splitext(dir_i)
    inBaseName = os.path.basename(name)
    outputdirection = os.path.join(dirname, inBaseName + '.nrrd')           
    sitk.WriteImage(image, outputdirection)
    
dirname = os.path.join(inputroot + 'ImageSplitFF')
datalist = ReadFoldandSort(dirname)
for dir_i in datalist:
    im = sitk.ReadImage(dir_i)
    imArray = sitk.GetArrayFromImage(im)
    image = sitk.GetImageFromArray(imArray)
    image.SetOrigin(OriginSet)
    image.SetDirection(DirectiongSet)
    image.SetSpacing(SpacingSet)
    name, ext = os.path.splitext(dir_i)
    inBaseName = os.path.basename(name)
    outputdirection = os.path.join(dirname, inBaseName + '.nrrd')          
    sitk.WriteImage(image, outputdirection)

dirname = os.path.join(inputroot + 'ImageSplitT2S')
datalist = ReadFoldandSort(dirname)
for dir_i in datalist:
    im = sitk.ReadImage(dir_i)
    imArray = sitk.GetArrayFromImage(im)
    image = sitk.GetImageFromArray(imArray)
    image.SetOrigin(OriginSet)
    image.SetDirection(DirectiongSet)
    image.SetSpacing(SpacingSet)
    name, ext = os.path.splitext(dir_i)
    inBaseName = os.path.basename(name)
    outputdirection = os.path.join(dirname, inBaseName + '.nrrd')            
    sitk.WriteImage(image, outputdirection)       
    sitk.WriteImage(image, outputdirection) 