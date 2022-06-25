# -*- coding: utf-8 -*-
"""
Post Processing after Unet.
"""
import os
import argparse
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import utility.CompareThePreandtruth as CTP
from utility.LabelpostProcess import postProcessing

GTFILE='Label.txt'
#%%
def create_pred_list(pred_dir):
    pred_list = glob.glob(os.path.join(pred_dir, '**/*_prediction.nii.gz'))
    F_list = glob.glob(os.path.join(pred_dir, '**/*_F.nii.gz'))
    FF_list = glob.glob(os.path.join(pred_dir, '**/*_FF.nii.gz'))
    W_list = glob.glob(os.path.join(pred_dir, '**/*_W.nii.gz'))
    T2S_list = glob.glob(os.path.join(pred_dir, '**/*_T2S.nii.gz'))
    GT_list = glob.glob(os.path.join(pred_dir, '**/*_truth.nii.gz'))
    pred_list.sort()
    F_list.sort()
    FF_list.sort()
    W_list.sort()
    T2S_list.sort()
    GT_list.sort()
    print('{} Test Images'.format(len(pred_list)))

    with open(os.path.join(pred_dir, 'pred_label.txt'), 'w') as pred_list_file:
        file_content = "\n".join(pred_list)
        pred_list_file.write(file_content)

    with open(os.path.join(pred_dir, 'F.txt'), 'w') as F_list_file:
        file_content = "\n".join(F_list)
        F_list_file.write(file_content)

    with open(os.path.join(pred_dir, 'FF.txt'), 'w') as FF_list_file:
        file_content = "\n".join(FF_list)
        FF_list_file.write(file_content)

    with open(os.path.join(pred_dir, 'W.txt'), 'w') as W_list_file:
        file_content = "\n".join(W_list)
        W_list_file.write(file_content)

    with open(os.path.join(pred_dir, 'T2S.txt'), 'w') as T2S_list_file:
        file_content = "\n".join(T2S_list)
        T2S_list_file.write(file_content)
    
    with open(os.path.join(pred_dir, 'GT.txt'), 'w') as GT_list_file:
        file_content = "\n".join(GT_list)
        GT_list_file.write(file_content)
    pass

def WriteListtoFile(filelist, filename):
    with open(filename, 'w') as f:
        for i in filelist:
            f.write(i+'\n')
    return 1  

# evaluate the segmentation with dice score    
def diceComputing(inputRootDir, outputRootDir, threshold):        
    
    predictInput = os.path.join(inputRootDir, 'pred_label.txt')
    groundTruthInput = os.path.join(inputRootDir, 'GT.txt')

    predictOutput = os.path.join(outputRootDir, 'Pred3DMod')
    if not os.path.exists(predictOutput):
        os.mkdir(predictOutput)
    
    dicorestat = CTP.CompareThePreandTruth(predictInput, groundTruthInput)
    dicorestat.readPredictImagetoList()
    dicorestat.readgroundTruthtoList()
    labelList = dicorestat.predictModification(predictOutput, threshold)
    WriteListtoFile(labelList, predictOutput + '/FileList.txt')
    dicorestat.diceScoreStatistics()
    
# evaluate the segmentation after filting with dice score 
def labelFiltering(inputRootDir, outputRootDir, threshold):
    predictOutput = os.path.join(outputRootDir, 'Pred3DMod')
    filepathLabel = os.path.join(predictOutput, 'FileList.txt')
    inputroot = inputRootDir
    outputroot = os.path.join(outputRootDir, 'Pred3D_FFilter')
    
    postProcessing(filepathLabel, inputroot, outputroot)

    predictInput = os.path.join(outputroot, 'FileList.txt')       
    groundTruthInput = os.path.join(inputRootDir, 'GT.txt')    
    predictOutput = os.path.join(outputRootDir, 'Pred3DMod_FFilter')
    if not os.path.exists(predictOutput):
        os.mkdir(predictOutput)
    dicorestat = CTP.CompareThePreandTruth(predictInput, groundTruthInput)
    dicorestat.readPredictImagetoList()
    dicorestat.readgroundTruthtoList()
    dicorestat.predictModification(predictOutput, threshold)    
    dicorestat.diceScoreStatistics()      

def main():
    parser = argparse.ArgumentParser(description = "BATNet command line tool")
    parser.add_argument("--input_folder", type=str, default = '../', help = "Dataset Folder")
    parser.add_argument("--output_folder", type=str, help = "project folder to save the output data.")
    parser.add_argument("--threshold", type=float, default=0.5,  help = "classification threshold.")
    args = parser.parse_args()
    if not os.path.exists(args.input_folder):
        os.mkdir(args.input_folder)
    create_pred_list(args.input_folder)   
    diceComputing(args.input_folder, args.output_folder, args.threshold)
    # employing filter to further improve the predicted label
    labelFiltering(args.input_folder, args.output_folder, args.threshold)

if __name__ == '__main__':
    main()
