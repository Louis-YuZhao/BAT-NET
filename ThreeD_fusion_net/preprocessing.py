import os
import sys
import argparse

sys.path.append('../')
from models.data import write_data_to_file, open_data_file
from models.data_prepare_v1 import get_training_and_validation_data
from models.data_split import GetListFromFiles, set_train_validation_test
from config import config
from train import set_seed


def fetch_training_data_files(folderNameDict):
    '''return list
    [(modality1, modality2, modality3, modality4, label)
    (modality1, modality2, modality3, modality4, label)]
    '''
    training_data_files = list()
    keys = list(folderNameDict.keys())    
    NN = 0
    for i in range(len(keys)):
        if i == 0:
            NN = len(folderNameDict[keys[i]])
        else:
            if NN != len(folderNameDict[keys[i]]):
                raise ValueError('check the len of the list with key: ' + keys[i])  
    for i in range(NN):
        subject_files = list()
        for modality in config["training_modalities"] + ["Label"]:
            subject_files.append(folderNameDict[modality][i])
        training_data_files.append(tuple(subject_files))
    return training_data_files


def data_preprocessing(input_data_root, overwrite=False):

   #-------------------------------------------------------#
    # convert input images into an hdf5 file
    outputfolder = os.path.dirname(config["data_file"])
    if overwrite or not os.path.exists(config["data_file"]):        
        train_path = {}
        train_path['FF'] = os.path.join(input_data_root, 'TrainingData/FF.txt')
        train_path['T2S'] = os.path.join(input_data_root, 'TrainingData/T2S.txt')
        train_path['F'] = os.path.join(input_data_root, 'TrainingData/F.txt')
        train_path['W'] = os.path.join(input_data_root, 'TrainingData/W.txt')
        train_path['pred_z'] = os.path.join(input_data_root, 'TrainingData/pred_z.txt')
        train_path['pred_x'] = os.path.join(input_data_root, 'TrainingData/pred_x.txt')
        train_path['pred_y'] = os.path.join(input_data_root, 'TrainingData/pred_y.txt')
        train_path['Label'] = os.path.join(input_data_root, 'TrainingData/Label.txt')
        
        test_path = {}
        test_path['FF'] = os.path.join(input_data_root, 'TestData/FF.txt')
        test_path['T2S'] = os.path.join(input_data_root, 'TestData/T2S.txt')
        test_path['F'] = os.path.join(input_data_root, 'TestData/F.txt')
        test_path['W'] = os.path.join(input_data_root, 'TestData/W.txt')
        test_path['pred_z'] = os.path.join(input_data_root, 'TestData/pred_z.txt')
        test_path['pred_x'] = os.path.join(input_data_root, 'TestData/pred_x.txt')
        test_path['pred_y'] = os.path.join(input_data_root, 'TestData/pred_y.txt')
        test_path['Label'] = os.path.join(input_data_root, 'TestData/Label.txt')     

        dataSplit = GetListFromFiles()
        folderName, trainNum, testNum = dataSplit.readImage(train_path, test_path)           
        
        training_files = fetch_training_data_files(folderName)
        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"])
        write_data_to_file(training_files, config["unnorm_data_file"], image_shape=config["image_shape"], ifUSENorm = False)
        data_file_opened = open_data_file(config["data_file"])
        #-------------------------------------------------------#

        #-------------------------------------------------------#
        # split the training data to training, validation, and test
        trainList = list(range(trainNum))
        testList = list(range(trainNum, trainNum+testNum))
        training_list, validation_list = set_train_validation_test(trainList, testList, t_v_split = config["validation_split"], 
                                training_file = config["training_file"], validation_file = config["validation_file"], 
                                test_file = config["test_file"], overwrite = overwrite)
        
        train_patch_data_dir, val_patch_data_dir = get_training_and_validation_data(data_file_opened, 
                                                                        outputfolder, 
                                                                        n_labels=config["n_labels"], 
                                                                        labels=config["labels"], 
                                                                        training_list = training_list,  
                                                                        validation_list = validation_list, 
                                                                        patch_shape = config["patch_shape"],
                                                                        training_patch_overlap = 0, 
                                                                        validation_patch_overlap = config["validation_patch_overlap"], 
                                                                        training_patch_start_offset = config["training_patch_start_offset"], 
                                                                        skip_blank = config["skip_blank"],
                                                                        unlabel=False)                        
        data_file_opened.close()

def main():
    #set seed
    set_seed(config['seed'])
    parser = argparse.ArgumentParser(description = "BATNet command line tool")
    parser.add_argument("--project-folder", type=str, help = "project folder to save the output data.")
    args = parser.parse_args()

    input_data_root = args.project_folder
    if not os.path.exists(os.path.join(input_data_root, "resultData")):
        os.mkdir(os.path.join(input_data_root, "resultData"))
    data_preprocessing(input_data_root, overwrite=config["overwrite"])



if __name__ == "__main__":
        
    main()