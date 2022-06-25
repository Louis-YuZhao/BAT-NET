import os

config = dict()
config["seed"] = 1025 #set seed for reproduction
config["image_shape"] = (288, 288, 96)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = (64, 64, 32)  # switch to None to train on the whole image
config["labels"] = (1,)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["FF", "T2S", "F", "W", "pred_z", "pred_x", "pred_y"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]

config["n_base_filters"] = 32


config["n_epochs"] = 10  # cutoff the training after this many epochs
config["initial_learning_rate"] = 1e-4
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["learning_rate_patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 20  # training will be stopped after this many epochs without the validation loss improving

config["batch_size"] = 8
config["validation_batch_size"] = 12
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 6  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["data_file"] = os.path.abspath("../resultData"+"/BAT_data.h5")
config["model_file"] = os.path.abspath("../resultData" +"/BAT_Combined_model.h5")
config["training_file"] = os.path.abspath("../resultData" +  "/BAT_Combined_training_ids.pkl")
config["validation_file"] = os.path.abspath("../resultData" +"/BAT_Combined_validation_ids.pkl")
config["test_file"] = os.path.abspath("../resultData" +"/BAT_Combined_test_ids.pkl")
config["trainingLog"] = os.path.abspath("../resultData"+"/BAT_Combined.log")
config["overwrite"] = True  # If True, will previous files. If False, will use previously written files.