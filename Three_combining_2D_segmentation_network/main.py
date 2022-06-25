from __future__ import print_function

from config import config
import os
os.environ["CUDA_VISIBLE_DEVICES"]= config['gpu']
import numpy as np
import argparse
from Model.Unet_model import get_unet
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

#%%
if config['dataDim'] == 'z':
    img_rows = config['dim_x']
    img_cols = config['dim_y']
    sliceNum = config['dim_z']
elif config['dataDim'] == 'x':
    img_rows = config['dim_z']
    img_cols = config['dim_y']
    sliceNum = config['dim_x']
elif config['dataDim'] == 'y': 
    img_rows = config['dim_z']
    img_cols = config['dim_x']
    sliceNum = config['dim_y']
else:
    raise ValueError ('DataDim should be z, x, y.')

overwrite = False
#%%
def train_and_predict(tempStore, model_input_channel, mode, learning_rate, batch_size, epochs, weight_dir, overwrite):

    if mode == 'train' or mode=='train_and_test':
        #---------------------------------#
        # trainging phase
        #---------------------------------#

        print('-'*30)
        print('Loading and preprocessing train data...')
        print('-'*30) 
            
        imgs_train = np.load(os.path.join(tempStore,'imgs_volume_train.npy'))
        imgs_label_train = np.load(os.path.join(tempStore,'imgs_label_train.npy'))
        
        print('-'*30)
        print('Creating and compiling model...')
        print('-'*30)
        #---------------------------------#
        model = get_unet(img_rows, img_cols, model_input_channel, learning_rate)
        #---------------------------------#
        if weight_dir:
            weightDir = weight_dir
        else:
            weightDir = os.path.join(tempStore, 'weights.h5')
        if (not overwrite) and (os.path.exists(weightDir)):
            model.load_weights(weightDir)

        model_checkpoint = ModelCheckpoint(weightDir, monitor='val_loss', save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto')
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        train_history = model.fit(imgs_train, 
                                  imgs_label_train, 
                                  batch_size=batch_size, 
                                  epochs=epochs,
                                  verbose=1, 
                                  shuffle=True, 
                                  validation_split=0.2, 
                                  callbacks=[model_checkpoint, early_stop]
                            )
        
        loss = train_history.history['loss']
        val_loss = train_history.history['val_loss']
        np.save(os.path.join(tempStore,'loss.npy'),loss)
        np.save(os.path.join(tempStore,'val_loss.npy'),val_loss)

    elif mode == 'test' or mode=='train_and_test':
        #---------------------------------#
        # test phase
        #---------------------------------#

        print('-'*30)
        print('Loading and preprocessing test data...')
        print('-'*30)
        
        imgs_test = np.load(os.path.join(tempStore,'imgs_volume_test.npy'))

        print('-'*30)
        print('Loading saved weights...')
        print('-'*30)
        model = get_unet(img_rows, img_cols, model_input_channel, learning_rate)
        if weight_dir:
            weightDir = weight_dir
        else:
            weightDir = os.path.join(tempStore, 'weights.h5')
        model.load_weights(weightDir)
        feature_medel = Model(inputs=model.input, outputs=model.get_layer('outputCov').output)

        print('-'*30)
        print('Predicting masks on test data...')
        print('-'*30)
        imgs_label_test = model.predict(imgs_test, verbose=1)
        imgs_feature_test = feature_medel.predict(imgs_test, verbose=1)
        np.save(os.path.join(tempStore,'imgs_label_test.npy'), imgs_label_test)
        np.save(os.path.join(tempStore,'imgs_feature_test.npy'), imgs_feature_test)

def main():
   
    parser = argparse.ArgumentParser(description = "BATNet command line tool")
    parser.add_argument("--project_folder", type=str, help = "project folder to save the output data.")
    parser.add_argument("--model_input_channel", type=int, default=4, help = "model input channel number")
    parser.add_argument("--mode", type=str, default='train_and_test', help = "model phase: train, test, or train_and_test")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--epochs", type=int, default=200, help = "training epochs")
    parser.add_argument("--batch_size", type=int, default=30, help = "class number")
    parser.add_argument("--overwrite", help = "Whether oeverwrite the model weight.", action = 'store_true')
    parser.add_argument("--weight_dir", type=str, default=None, help = "the path to save the model weights")    
    args = parser.parse_args()
    
    tempStore = os.path.join(args.project_folder, config['dataDim'], 'tempData')
    train_and_predict(tempStore, 
                      model_input_channel = args.model_input_channel,
                      mode = args.mode,
                      learning_rate = args.learning_rate,
                      batch_size = args.epochs,
                      epochs = args.batch_size,
                      weight_dir = args.weight_dir,
                      overwrite = args.overwrite
                )

if __name__ == '__main__':
    main()
