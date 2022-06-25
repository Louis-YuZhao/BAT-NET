import os
import argparse
from config import config
from models.prediction import run_validation_case,trans_validation_case
from models.utils import pickle_load

#%%
def validation(rootdir, model_path):
    prediction_dir = rootdir
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    validation_indices = pickle_load(config["validation_file"])
    for i in range(len(validation_indices)):
        run_validation_case(test_index=i, 
                            out_dir=os.path.join(prediction_dir, "case_{}".format(i)),
                            model_file=model_path, 
                            validation_keys_file=config["validation_file"],
                            training_modalities=config["training_modalities"], 
                            output_label_map=True,
                            labels=config["labels"], 
                            hdf5_file=config["data_file"])
        
        trans_validation_case(test_index=i, 
                              out_dir=os.path.join(prediction_dir, "case_{}".format(i)), 
                              hdf5_file=config["unnorm_data_file"], 
                              validation_keys_file =config["validation_file"], 
                              training_modalities=config["training_modalities"])

def test(rootdir, model_path):
    prediction_dir = rootdir
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    test_indices = pickle_load(config["test_file"])
    for i in range(len(test_indices)):
        run_validation_case(test_index=i, 
                            out_dir=os.path.join(prediction_dir, "case_{}".format(i)),
                            model_file=model_path, 
                            validation_keys_file=config["test_file"],
                            training_modalities=config["training_modalities"], 
                            output_label_map=True,
                            labels=config["labels"], 
                            hdf5_file=config["data_file"])

        trans_validation_case(test_index=i, 
                              out_dir=os.path.join(prediction_dir, "case_{}".format(i)), 
                              hdf5_file=config["unnorm_data_file"], 
                              validation_keys_file =config["test_file"], 
                              training_modalities=config["training_modalities"])

def train_validation(rootdir, model_path):
    prediction_dir = os.path.join(rootdir,'train_prediction')
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    validation_indices = pickle_load(config["training_file"])
    for i in range(len(validation_indices)):
        run_validation_case(test_index=i, 
                            out_dir=os.path.join(prediction_dir, "case_{}".format(i)),
                            model_file=model_path, 
                            validation_keys_file=config["training_file"],
                            training_modalities=config["training_modalities"], 
                            output_label_map=True,
                            labels=config["labels"], 
                            hdf5_file=config["data_file"])
        
        trans_validation_case(test_index=i, 
                              out_dir=os.path.join(prediction_dir, "case_{}".format(i)), 
                              hdf5_file=config["unnorm_data_file"], 
                              validation_keys_file =config["training_file"], 
                              training_modalities=config["training_modalities"])

def main():
    parser = argparse.ArgumentParser(description = "BATNet command line tool")
    parser.add_argument("--output_folder", type=str, help = "folder to restore the outputs")
    args = parser.parse_args()
    test(rootdir=args.output_folder,
         model_path=config["model_file"]
    )
if __name__ == "__main__":
    main()
