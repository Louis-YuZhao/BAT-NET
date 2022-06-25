#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
main function for Brown Adipose Tissue segmentation

Created on Fri Sep  8 08:26:03 2017

@author: louis
"""

import numpy as np
import os
import libPreProcess as pp


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

def FFprocessing():
    '''
    whole pipline for multi atlas segmentation
    '''    
    # preprocessing
    IFLogOutput = True
    showTempImage = False
    inputroot = '/media/data/louis/ResearchData/BrownAdiposeTissue/ImageSplitNrrd'
    outputroot = '/media/data/louis/ResearchData/BrownAdiposeTissue/ImageSplitNrrd'
    outputdir = outputroot + '/FF_PRE'
    
    filepathFF = inputroot + '/FF/FileList.txt' 
    filepathT2Star = inputroot + '/T2S/FileList.txt' 
    filepathFat = inputroot + '/F/FileList.txt'
    ilepathWater = inputroot + '/W/FileList.txt'
    
    listFF = readTxtIntoList(filepathFF)
    listT2Star = readTxtIntoList(filepathT2Star)
    listFat = readTxtIntoList(filepathFat)
    listWater = readTxtIntoList(ilepathWater)   
    
    if len(listFF) == len(listT2Star) == len(listFat) == len(listWater):
        N = len(listFF)
    else:
        raise ValueError('the length of the files should be same')
    
    # PreProcessing WF image   
    FFprocessing = pp.BATpreprocessingFF(outputdir, showTempImage)
    for i in range(N):
        FFprocessing.inputImage(listFF[i], listT2Star[i], listFat[i], listWater[i])
        FFprocessing.removeDataByThreshold((0,100), ifLogOutput = False)
        FFprocessing.removeBackgroundmap(threshold = 100, ifLogOutput = False)
        FFprocessing.removeSkinVoxels(FFthreshold = 45, T2Sthreshold = 10, iternum = 2,CCthreshold = 1,\
                                      filterSize = (3,3), ClossingSize = np.ones((3,3)),\
                                      ErosionSize = np.ones((7,7)), ifLogOutput = IFLogOutput)
        print "processing num: %d" %(i)
    # collecting the preprocessed result
    ImageList = []
    for root, dirnames, filenames in os.walk(outputdir):
        for name in filenames:
            if name.endswith('_WF_RBG.nrrd'):
                ImageDir = os.path.join(root, name)
                ImageList.append(ImageDir)
    ImageList.sort()
    WriteListtoFile(ImageList, outputdir + '/FileList.txt')

def T2Sprocessing():
    '''
    whole pipline for multi atlas segmentation
    '''    
    # preprocessing
    IFLogOutput = True
    inputroot = '/media/data/louis/ResearchData/BrownAdiposeTissue/ImageSplitNrrd'
    outputroot = '/media/data/louis/ResearchData/BrownAdiposeTissue/ImageSplitNrrd'
    outputdir = outputroot + '/T2S_PRE'
    
    filepathFF = inputroot + '/FF/FileList.txt' 
    filepathT2Star = inputroot + '/T2S/FileList.txt' 
    filepathFat = inputroot + '/F/FileList.txt'
    ilepathWater = inputroot + '/W/FileList.txt'
    
    listFF = readTxtIntoList(filepathFF)
    listT2Star = readTxtIntoList(filepathT2Star)
    listFat = readTxtIntoList(filepathFat)
    listWater = readTxtIntoList(ilepathWater)   
    
    if len(listFF) == len(listT2Star) == len(listFat) == len(listWater):
        N = len(listFF)
    else:
        raise ValueError('the length of the files should be same')
    
    # PreProcessing WF image   
    T2Spro = pp.BATpreprocessingT2S(outputdir)
    for i in range(N):
        T2Spro.inputImage(listFF[i], listT2Star[i], listFat[i], listWater[i])
        T2Spro.removeBackgroundmap(threshold = 100, ifLogOutput = IFLogOutput)
        # T2Spro.removeSkinVoxels(FFthreshold = 45, T2Sthreshold = 10, iternum = 2,CCthreshold = 1,\
        #                               filterSize = (3,3), ClossingSize = np.ones((3,3)),\
        #                               ErosionSize = np.ones((7,7)), ifLogOutput = IFLogOutput)
        T2Spro.reduceNoise(filterSize = (3,3,3), ifLogOutput = IFLogOutput)
        print "processing num: %d" %(i)
    # collecting the preprocessed result
    ImageList = []
    for root, dirnames, filenames in os.walk(outputdir):
        for name in filenames:
            if name.endswith('_T2S_RBG.nrrd'):
                ImageDir = os.path.join(root, name)
                ImageList.append(ImageDir)
    ImageList.sort()
    WriteListtoFile(ImageList, outputdir + '/FileList.txt')

    
if __name__ == "__main__":
    FFprocessing()
#    T2Sprocessing()