from __future__ import print_function

import os
import argparse
import numpy as np
import subprocess
from utility.libDataPrepare import dataPrepare
from utility.libDataPrepare import DatasplitFile
from config import config


def main():


    parser = argparse.ArgumentParser(description = "BATNet command line tool")
    parser.add_argument("--image-data-directory", type=str, help = "directory of data.")
    parser.add_argument("--project-folder", type=str, help = "project folder to save the output data.")
    args = parser.parse_args()

    image_data_root = args.image_data_directory
    data_root = args.project_folder
    tempStore = os.path.join(data_root, config['dataDim'], 'tempData')
    if not os.path.exists(tempStore):
        subprocess.call('mkdir ' + '-p ' + tempStore, shell=True)


    # Data used in the network 
    data_path = {}
    data_path['FF'] = os.path.join(image_data_root, 'FF_pre')
    data_path['T2S'] = os.path.join(image_data_root, 'T2S_pre')
    data_path['F'] = os.path.join(image_data_root, 'F')
    data_path['W'] = os.path.join(image_data_root, 'W')
    data_path['Label'] = os.path.join(image_data_root, 'Label_processed')


    if config['divideData'] == True:
        # splited the data into training and test group
        train_path, test_path = DatasplitFile(data_path, config['test_num'],outputDir=data_root)  
    else:            
        # manually determine the training and test group
        train_path = {}
        train_path['FF'] = os.path.join(data_root, 'TrainingData/FF.txt')
        train_path['T2S'] = os.path.join(data_root, 'TrainingData/T2S.txt')
        train_path['F'] = os.path.join(data_root, 'TrainingData/F.txt')
        train_path['W'] = os.path.join(data_root, 'TrainingData/W.txt')
        train_path['Label'] = os.path.join(data_root, 'TrainingData/Label.txt')
        
        test_path = {}
        test_path['FF'] = os.path.join(data_root, 'TestData/FF.txt')
        test_path['T2S'] = os.path.join(data_root, 'TestData/T2S.txt')
        test_path['F'] = os.path.join(data_root, 'TestData/F.txt')
        test_path['W'] = os.path.join(data_root, 'TestData/W.txt')


    batData = dataPrepare(config)    
    batData.MultiChannelDataPrepare(train_path, test_path, tempStore)
    batData.Train_and_predict_dataPrepare(tempStore)
    batData.del_train_data(tempStore)
    batData.del_test_data(tempStore)

    pass

    
if __name__ == '__main__':
        
    main()