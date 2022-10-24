from __future__ import print_function

from config import config
import os
os.environ["CUDA_VISIBLE_DEVICES"]= config['gpu']
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

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

smooth = 1.
UnetChannelNum = 4
overwrite = True
learningRate = config['learningRate']
batch_size = config['batch_size']
epochs = config['epochs']

ALPHA = config['Tversky_alpha']
BETA = 1 - ALPHA
GAMMA = config['focal_gamma']
ALPHA_F = config['focal_alpha']

#%%
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def FocalLoss(targets, inputs, alpha=ALPHA_F, gamma=GAMMA):    
    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss

def FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))
            
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    FocalTversky = K.pow((1-Tversky), gamma)
    
    return FocalTversky

def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
        
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))
    
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    
    return 1 - Tversky