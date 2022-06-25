#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Set the Origin, Direction, and Spacing

Created on 10.12.2018

@author: louis
"""

import os
import SimpleITK as sitk

OriginSet = [-198.2,-159.6,-79.04]
DirectiongSet = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
SpacingSet = (1, 1, 1)

def ReadFoldandSort(data_path, ext='.nrrd'):
    imageList = []
    for fileItem in os.listdir(data_path):
        if fileItem.endswith(ext):
            imageList.append(os.path.join(data_path, fileItem))
    imageList.sort()
    return imageList

#%%

def main(inputroot):   

    dirname = inputroot
    datalist = ReadFoldandSort(dirname)

    for dir_i in datalist:
        image = sitk.ReadImage(dir_i)
        image.SetOrigin(OriginSet)
        image.SetDirection(DirectiongSet)
        image.SetSpacing(SpacingSet)
        name, ext = os.path.splitext(dir_i)
        inBaseName = os.path.basename(name)
        outputdirection = os.path.join(dirname, inBaseName + '.nrrd')             
        sitk.WriteImage(image, outputdirection)
        print(inBaseName)
        
if __name__ == "__main__":
    root = '/media/data/louis/ResearchData/BrownAdiposeTissue/'
#    folderlist = ['F', 'FF_PRE','Labels','rawLabels','wholeBodyMask','T2S_PRE', 'W']
    folderlist = ['manualLabel']
    for folder in folderlist:
        inputroot = os.path.join(root, folder)
        main(inputroot)