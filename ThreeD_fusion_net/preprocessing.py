import os
import sys
import argparse
sys.path.append('../')
from models.data import write_data_to_file, open_data_file
from models.data_prepare_v1 import get_training_and_validation_data
from models.data_split import GetListFromFiles, set_train_validation_test
from config import config
from train import set_seed
from train import fetch_training_data_files

def main():
    parser = argparse.ArgumentParser(description = "BATNet command line tool")
    parser.add_argument("--input_data_root", type=str, help = "input data restored folder")
    args = parser.parse_args()
    
    #set seed
    set_seed(config['seed'])

    os.makedirs(os.path.dirname(config["data_file"]), exist_ok=True)    
    
    input_data_root = args.input_data_root
    overwrite=config["overwrite"]

   #-------------------------------------------------------#
    # convert input images into an hdf5 file
    outputfolder = os.path.dirname(config["data_file"])
    
    if overwrite or not os.path.exists(config["data_file"]):        
        train_path = {}
        train_path['FF'] = os.path.join(input_data_root, 'TrainingData/FF/Filelist.txt')
        train_path['T2S'] = os.path.join(input_data_root, 'TrainingData/T2S/Filelist.txt')
        train_path['F'] = os.path.join(input_data_root, 'TrainingData/F/Filelist.txt')
        train_path['W'] = os.path.join(input_data_root, 'TrainingData/W/Filelist.txt')
        train_path['pred_z'] = os.path.join(input_data_root, 'TrainingData/pred_z/Filelist.txt')
        train_path['pred_x'] = os.path.join(input_data_root, 'TrainingData/pred_x/Filelist.txt')
        train_path['pred_y'] = os.path.join(input_data_root, 'TrainingData/pred_y/Filelist.txt')
        train_path['Label'] = os.path.join(input_data_root, 'TrainingData/Label/Filelist.txt')
        
        test_path = {}
        test_path['FF'] = os.path.join(input_data_root, 'TestData/FF/Filelist.txt')
        test_path['T2S'] = os.path.join(input_data_root, 'TestData/T2S/Filelist.txt')
        test_path['F'] = os.path.join(input_data_root, 'TestData/F/Filelist.txt')
        test_path['W'] = os.path.join(input_data_root, 'TestData/W/Filelist.txt')
        test_path['pred_z'] = os.path.join(input_data_root, 'TestData/pred_z/Filelist.txt')
        test_path['pred_x'] = os.path.join(input_data_root, 'TestData/pred_x/Filelist.txt')
        test_path['pred_y'] = os.path.join(input_data_root, 'TestData/pred_y/Filelist.txt')
        test_path['Label'] = os.path.join(input_data_root, 'TestData/Label/Filelist.txt')     

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

if __name__ == "__main__":
    main()
