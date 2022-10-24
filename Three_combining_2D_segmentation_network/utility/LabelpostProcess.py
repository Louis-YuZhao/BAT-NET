#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
main function for Brown Adipose Tissue segmentation

Created on Fri Sep  8 08:26:03 2017

@author: louis
"""

import numpy as np
import os
import string
import subprocess



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

def postProcessing(filepathLabel, inputroot, outputroot):    
    #%%
    # PostProcessing 
    """
    filePathLabel: obtained label path
    inputRoot: the paths of W, F, FF, and T2S folders
    outputRoot: the output folder of the addressed data
    """
     
    refListLabel = readTxtIntoList(filepathLabel)
    refListLabel.sort()

    refListFF = readTxtIntoList(os.path.join(inputroot,'FF.txt'))
    refListFF.sort()
    refListT2Star = readTxtIntoList(os.path.join(inputroot,'T2S.txt'))
    refListT2Star.sort()
    refListFat = readTxtIntoList(os.path.join(inputroot,'F.txt'))
    refListFat.sort()
    refListWater = readTxtIntoList(os.path.join(inputroot,'W.txt'))
    refListWater.sort() 
   
    IFLogOutput = True
    
    if len(refListLabel)==len(refListFF)==len(refListT2Star)==len(refListFat)==len(refListWater):
        N = len(refListFF)
    else:
        raise ValueError('the length of the files should be same')
    
    for i in xrange(len(refListLabel)):
        item = refListLabel[i]
        filename = os.path.basename(item)
        checkItem = string.join((filename).split("_")[1:2], "_")
        if not ((checkItem in refListFF[i]) and (checkItem in refListT2Star[i])\
            and (checkItem in refListFat[i]) and (checkItem in refListWater[i])):
            raise ValueError(str(i)+' th do not match each other')
    
    dilationPara = {}
    dilationPara['IfDilation'] = True
    dilationPara['structure'] = np.ones((3,3,3))
    dilationPara['iterations'] = 1
    ClossingPara = {}
    ClossingPara['IfClossing'] = True
    ClossingPara['structure'] = np.ones((3,3,3))
    ClossingPara['iterations'] = 1
    ErosionPara = {}
    ErosionPara['IfErosion'] = True
    ErosionPara['structure'] = np.ones((3,3,3))
    ErosionPara['iterations'] = 1
    removeSmallObj = {}
    removeSmallObj['min_size'] = 10
    removeSmallObj['connectivity'] = 4    
    
    refinedLabeldir = outputroot
    if not os.path.exists(refinedLabeldir):
        subprocess.call('mkdir ' + '-p ' + refinedLabeldir, shell=True)
    
    filelist =[]
    for i in xrange(N):
        postProcessing = pr.postRefinement(refinedLabeldir)
        postProcessing.inputImage(refListLabel[i], refListFF[i], refListT2Star[i],\
                                  refListFat[i], refListWater[i])
        postProcessing.fineAdjustSegmentation(FFthresholdLow = 45, FFthresholdHigh=100,\
                                              T2Sthreshold = 10, ifLogOutput = IFLogOutput)
        filelist.append(postProcessing.removeBoneadnAir(threshold = 300, ifLogOutput = IFLogOutput))
        postProcessing.finalRefine(dilationPara, ClossingPara, ErosionPara,\
                                   removeSmallObj, ifLogOutput = IFLogOutput)
        print("processing num: %d" %(i))
    
    filename = os.path.join(outputroot,'FileList.txt')
    WriteListtoFile(filelist, filename)
    