from __future__ import print_function
from cgi import test

from config import config
import os
os.environ["CUDA_VISIBLE_DEVICES"]= config['gpu']
import argparse
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from model.unet_model import get_unet

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

if config['dataDim'] == 'z':
    img_rows = config['dim_x']
    img_cols = config['dim_y']
elif config['dataDim'] == 'x':
    img_rows = config['dim_z']
    img_cols = config['dim_y']
elif config['dataDim'] == 'y': 
    img_rows = config['dim_z']
    img_cols = config['dim_x']
else:
    raise ValueError ('DataDim should be z, x, y.')

UnetChannelNum = 4
learningRate = config['learningRate']
batch_size = config['batch_size']
epochs = config['epochs']


def train_and_predict(tempStore, mode, overwrite):

    if mode == 'train' or mode == 'train_and_test':

        #---------------------------------#
        # training phase
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
        model = get_unet(img_rows, img_cols, UnetChannelNum)
        trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

        
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))
        
        #---------------------------------#
        weightDir = os.path.join(tempStore, 'weights.h5')
        if (not overwrite) and (os.path.exists(weightDir)):
            model.load_weights(weightDir)

        model_checkpoint = ModelCheckpoint(weightDir, monitor='val_loss', save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        train_history = model.fit(imgs_train, imgs_label_train, batch_size=batch_size, epochs=epochs,\
        verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint, early_stop])
        
        loss = train_history.history['loss']
        val_loss = train_history.history['val_loss']
        np.save(os.path.join(tempStore,'loss.npy'),loss)
        np.save(os.path.join(tempStore,'val_loss.npy'),val_loss)
        imgs_predict_train = model.predict(imgs_train, verbose=1)
        np.save(os.path.join(tempStore,'imgs_predict_train.npy'), imgs_predict_train)

    if mode == 'test' or mode == 'train_and_test':
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
        model = get_unet(img_rows, img_cols, UnetChannelNum)
        model.load_weights(os.path.join(tempStore, 'weights.h5'))
        feature_medel = Model(inputs=model.input,outputs=model.get_layer('outputCov').output)

        print('-'*30)
        print('Predicting masks on test data...')
        print('-'*30)
        imgs_label_test = model.predict(imgs_test, verbose=1)
        np.save(os.path.join(tempStore,'imgs_label_test.npy'), imgs_label_test)
    

def main():

    parser = argparse.ArgumentParser(description = "BATNet command line tool")
    parser.add_argument("--project-folder", type=str, help = "project folder to save the output data.")
    parser.add_argument("--mode", type=str, default='train_and_test', help = "model phase: train, test, or train_and_test")
    parser.add_argument("--overwrite", help = "Whether oeverwrite the model weight.", action = 'store_true')
    args = parser.parse_args()
    
    tempStore = os.path.join(args.project_folder, config['dataDim'], 'tempData')
    train_and_predict(tempStore, args.mode, args.overwrite)

if __name__ == '__main__':
    main()
