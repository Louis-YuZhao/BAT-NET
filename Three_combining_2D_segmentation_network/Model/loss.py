from config import config
import os
os.environ["CUDA_VISIBLE_DEVICES"]= config['gpu']
import numpy as np
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

#%%
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

