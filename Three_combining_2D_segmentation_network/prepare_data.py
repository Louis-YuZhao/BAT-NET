#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import argparse
import subprocess
from utility.libDataPrepare import dataPrepare
from utility.ImageAdjust import imageAdjust
from utility.libDataPrepare import DatasplitFile
from config import config

#%%
def main():
    parser = argparse.ArgumentParser(description = "BATNet command line tool")
    parser.add_argument("--data_folder", type=str, default = '../', help = "Dataset Folder")
    parser.add_argument("--project_folder", type=str, default='../', help = "Path to resotre the processed results")
    args = parser.parse_args()

    data_folder = args.data_folder
    project_folder = os.path.join(args.project_folder, config['dataDim'], 'tempData')
    if not os.path.exists(project_folder):
        subprocess.call('mkdir ' + '-p ' + project_folder, shell=True)
   
    # Data used in the network 
    data_path = {}
    data_path['FF'] = os.path.join(data_folder, 'data/FF/')
    data_path['T2S'] = os.path.join(data_folder, 'data/T2S/')
    data_path['F'] = os.path.join(data_folder, 'data/F/')
    data_path['W'] = os.path.join(data_folder, 'data/W/')
    data_path['Label'] = os.path.join(data_folder, 'data/Labels/')
    # data_path['Mask'] = os.path.join(data_folder, 'data/wholeBodyMask/')

    if config['radomly_split_the_data'] == True:
        # splited the data into training and test group
        train_path, test_path = DatasplitFile(data_path, config['test_num'], outputDir=data_folder)  
    else:            
        # manually determine the training and test group
        train_path = {}
        train_path['FF'] = os.path.join(data_folder, 'TrainingData/FF.txt')
        train_path['T2S'] = os.path.join(data_folder, 'TrainingData/T2S.txt')
        train_path['F'] = os.path.join(data_folder, 'TrainingData/F.txt')
        train_path['W'] = os.path.join(data_folder, 'TrainingData/W.txt')
        train_path['Label'] = os.path.join(data_folder, 'TrainingData/Label.txt')
        test_path = {}
        test_path['FF'] = os.path.join(data_folder, 'TestData/FF.txt')
        test_path['T2S'] = os.path.join(data_folder, 'TestData/T2S.txt')
        test_path['F'] = os.path.join(data_folder, 'TestData/F.txt')
        test_path['W'] = os.path.join(data_folder, 'TestData/W.txt')    

    batData = dataPrepare(config)    
    batData.MultiChannelDataPrepare(train_path, test_path, project_folder)
    batData.Train_and_predict_dataPrepare(project_folder)
    batData.del_train_data(project_folder)
    batData.del_test_data(project_folder)

if __name__ == '__main__':
    main()