# -*- coding: utf-8 -*-
"""
class for adjusting the image size 
"""

import os
import numpy as np
import SimpleITK as sitk 
import subprocess
from scipy.misc import imresize

class imageAdjust(object):
    def __init__(self,inputFolderName, outputFolderName, config):
        self.inputFolderName = inputFolderName
        self.outputFolderName = outputFolderName
        self.config = config

    def ReadFoldandSort(self, fileEnd=".nrrd"):
        imageList = []
        for fileItem in os.listdir(self.inputFolderName):
            if fileItem.endswith(fileEnd):
                imageList.append(os.path.join(self.inputFolderName, fileItem))
        imageList.sort()
        self.FileList = imageList
        return self.FileList

    def imagePadding(self, newsize, config):

        fileList = []
        z_pad = newsize['dim_z']
        x_pad = newsize['dim_x']
        y_pad = newsize['dim_y']

        for imagedir in self.FileList:
            image = sitk.ReadImage(imagedir)
            imageArray = sitk.GetArrayFromImage(image)
            z_org, x_org, y_org = np.shape(imageArray)
            
            z_dif = z_pad - z_org
            x_dif = x_pad - x_org
            y_dif = y_pad - y_org

            z_dif_sta = z_dif/int(2)
            z_dif_end = z_dif - z_dif_sta

            x_dif_sta = x_dif/int(2)
            x_dif_end = x_dif - x_dif_sta

            y_dif_sta = y_dif/int(2)
            y_dif_end = y_dif - y_dif_sta

            pad_para = ((z_dif_sta,z_dif_end),(x_dif_sta,x_dif_end),(y_dif_sta,y_dif_end))
            imageArrayPad = np.pad(imageArray, pad_para,'constant', constant_values=0)

            img = sitk.GetImageFromArray(imageArrayPad)
            img.SetOrigin(image.GetOrigin())
            img.SetSpacing(image.GetSpacing())
            img.SetDirection(image.GetDirection())            

            baseName = os.path.basename(imagedir)
            paddedOurputDir = os.path.join(self.outputFolderName, 'ImagePadded')
            if not os.path.exists(paddedOurputDir):
                subprocess.call('mkdir ' + '-p ' + paddedOurputDir, shell=True)
            fn = os.path.join(self.outputFolderName, 'ImagePadded', baseName)
            fileList.append(fn)
            sitk.WriteImage(img,fn)
            
        self.paddedFileList = fileList
        return self.paddedFileList

    def imageResizing(self, newsize, config):

        fileList = []
        z_resize = newsize['dim_z']
        x_resize = newsize['dim_x']
        y_resize = newsize['dim_y']

        for imagedir in self.FileList:
            image = sitk.ReadImage(imagedir)
            imageArray = sitk.GetArrayFromImage(image)
            z_org, x_org, y_org = np.shape(imageArray)

            imageArrayResize = imresize(imageArray, size=(z_resize,x_resize,y_resize),interp=config['interpMethod'])

            img = sitk.GetImageFromArray(imageArrayResize)
            img.SetOrigin(image.GetOrigin())
            img.SetSpacing(image.GetSpacing())
            img.SetDirection(image.GetDirection())            

            baseName = os.path.basename(imagedir)
            resizedOurputDir = os.path.join(self.outputFolderName, 'ImageResized')
            if not os.path.exists(resizedOurputDir):
                subprocess.call('mkdir ' + '-p ' + resizedOurputDir, shell=True)
            fn = os.path.join(self.outputFolderName, 'ImageResized', baseName)
            fileList.append(fn)
            sitk.WriteImage(img,fn)
        
        self.resizedFileList = fileList
        return self.resizedFileList