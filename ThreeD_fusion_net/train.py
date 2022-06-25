import os
import sys
import argparse
import numpy as np
sys.path.append('../')

from config import config
from models.data import open_data_file
from models.data_prepare_v1 import get_training_and_validation_generators
from models.model import combineNet_3d
from models.training import load_old_model, train_model

from models.GPU_config import gpuConfig
os.environ["CUDA_VISIBLE_DEVICES"]= gpuConfig['GPU_using']

#%%
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    print('Random Seed: {}'.format(seed))
    
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

def main():
    parser = argparse.ArgumentParser(description = "BATNet command line tool")
    parser.add_argument('--learning_rate', type=float, default=None, help='learning rate')
    parser.add_argument("--epochs", type=int, default=None, help = "training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help = "class number") 
    args = parser.parse_args()

    if args.learning_rate:
        config["initial_learning_rate"] = args.learning_rate
    if args.epochs:
        config["n_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    #set seed
    set_seed(config['seed'])
    overwrite=config["overwrite"]
    #-------------------------------------------------------#
    # convert input images into an hdf5 file
    outputfolder = os.path.dirname(config["data_file"])
    train_patch_data_dir = os.path.join(outputfolder,'train_patch_data_save.h5')
    val_patch_data_dir = os.path.join(outputfolder,'val_patch_data_save.h5')
    train_patch_data_file = open_data_file(train_patch_data_dir)
    val_patch_data_file = open_data_file(val_patch_data_dir)

    #-------------------------------------------------------#
    # set the model
    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = combineNet_3d(input_shape=config["input_shape"], 
                                n_labels=config["n_labels"], 
                                depth=5, 
                                n_base_filters=config["n_base_filters"],
                                normMethod = 'batch_norm',
                                initial_learning_rate = config["initial_learning_rate"])                                                                                                     
    
    #-------------------------------------------------------#
    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps \
    = get_training_and_validation_generators(train_patch_data_file, 
                                            val_patch_data_file, 
                                            batch_size = config["batch_size"],
                                            n_labels=config["n_labels"], 
                                            labels=config["labels"], 
                                            validation_batch_size = config["validation_batch_size"], 
                                            patch_shape = config["patch_shape"],
                                            augment = config["augment"], 
                                            augment_flip=config["flip"], 
                                            augment_distortion_factor=config["distort"],  
                                            permute=config["permute"])    
    #-------------------------------------------------------#
    # run training combineNet
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                n_epochs=config["n_epochs"],
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["learning_rate_patience"],
                early_stopping_patience=config["early_stop"],
                workers = 0, # when workers is not 0, it will cause the problem.
                use_multiprocessing=False, 
                logging_file = config["trainingLog"])    
    #-------------------------------------------------------#
    train_patch_data_file.close()
    val_patch_data_file.close()
    
if __name__ == "__main__":
    main()