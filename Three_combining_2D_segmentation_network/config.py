#!/usr/bin/env python2
# -*- coding: utf-8 -*-

config = dict()

# data split
config['divideData'] = False
config['test_num']= 10

config['org_dim_z']=96
config['org_dim_x']=288
config['org_dim_y']=288

config['dim_z']=96
config['dim_x']=288
config['dim_y']=288

config['dataDim']= 'x'
config['rounds'] = 5

# data preparing
# data propressing when loading the data
config['NormType'] = 0
# data normalization when preparing the data
config['IfglobalNorm'] = True

# parameters of the network
config['learningRate'] = 1e-3
config['batch_size']= 128
config['epochs'] = 250
config['gpu']= '0'

# parameters of losses
config['Tversky_alpha'] = 0.6
config['focal_gamma'] = 0.75
config['focal_alpha'] = 1.0
