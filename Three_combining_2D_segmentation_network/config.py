#!/usr/bin/env python2
# -*- coding: utf-8 -*-
config = dict()

# base line information 

# original image dimension
config['org_dim_z']=93
config['org_dim_x']=288
config['org_dim_y']=288

# the dimension after processing. 
config['dim_z']=96
config['dim_x']=288
config['dim_y']=288

# set the image parameters
Reference={}    
Reference['origin'] = (0, 0, 0)
Reference['spacing'] = (1, 1, 2)
Reference['direction'] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
config['Image_Reference'] = Reference

# data preparing

# data split in processing 
config['radomly_split_the_data']=False # Select true if one would like radomly re-split the dataset into a training set and a tes set.
config['test_num']=8 # Number of 

# data propressing when loading the data
config['NormType'] = 0
# data normalization when preparing the data
config['IfglobalNorm'] = True

# parameters during training the network
config['dataDim'] = 'y' # z: Axial x: Coronal y: Sagittal
config['channel'] = 4
config['gpu']='0'

# parameters used when evaluating the predicted result
config['GTFILE']='Label.txt'
config['preThreshold'] = 0.5