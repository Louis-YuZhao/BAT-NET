# -*- coding: utf-8 -*-
"""
Post Processing after Unet.
"""
import os
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utility.CompareThePreandtruth as CTP
from utility.LabelpostProcess import postProcessing
from config import config

GTFILE='Label.txt'
# GTFILE='ModifiedLabel.txt'
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

preThreshold = 0.5
#%%
def WriteListtoFile(filelist, filename):
    with open(filename, 'w') as f:
        for i in filelist:
            f.write(i+'\n')
    return 1
    
def VolumeDataToVolumes(ThreeDImageDir, Reference, sliceNum, image_rows, image_cols):

    imgs_label_test = np.load(os.path.join(tempStore,'imgs_label_test.npy'))
    dimZ, _, _, _ = np.shape(imgs_label_test)
    NumOfImage = int(dimZ/sliceNum)
    
    imgs_id_test = np.load(os.path.join('../', 'z', 'tempData', 'imgs_id_test.npy'))  # For convenience we take fix test id
    ImageList = []
    for i in range(NumOfImage):
        threeDImageArray = np.zeros((sliceNum, image_rows, image_cols))
        for j in range(sliceNum):
            threeDImageArray[j,:,:] = imgs_label_test[i*sliceNum+j,:,:,0]
        
        if config['dataDim'] == 'z':
            pass
        elif config['dataDim'] == 'x':
            threeDImageArray = np.swapaxes(threeDImageArray,0,1)
        elif config['dataDim'] == 'y': 
            threeDImageArray = np.swapaxes(threeDImageArray,0,1)
            threeDImageArray = np.swapaxes(threeDImageArray,1,2)
        else:
            raise ValueError ('DataDim should be z, x, y.')
        
        threeDimage = sitk.GetImageFromArray(threeDImageArray)
        threeDimage.SetOrigin(Reference['origin'])                               
        threeDimage.SetSpacing(Reference['spacing'])                                
        threeDimage.SetDirection(Reference['direction'])
        
        ThreeDImagePath = os.path.join(ThreeDImageDir, 'BAT_' + str(imgs_id_test[i]).split('_')[1]+'_pred.nii.gz')
        sitk.WriteImage(threeDimage, ThreeDImagePath)
        ImageList.append(ThreeDImagePath)

    WriteListtoFile(ImageList, ThreeDImageDir + '/FileList.txt')
    return ImageList
    
def Test3DDataCollecting(inputRootDir,outputRootDir, refImage, sliceNum, image_rows, image_cols):
    ThreeDImageDir = os.path.join(outputRootDir, 'Pred3D')
    if not os.path.exists(ThreeDImageDir):
        os.mkdir(ThreeDImageDir)    
    Reference={}
    refImage = sitk.ReadImage(refImage)
    Reference['origin'] = refImage.GetOrigin()
    Reference['spacing'] = refImage.GetSpacing()
    Reference['direction'] = refImage.GetDirection()
    VolumeDataToVolumes(ThreeDImageDir, Reference, sliceNum, image_rows, image_cols)

# evaluate the segmentation with dice score    
def diceComputing(inputRootDir,outputRootDir):        
    ThreeDImageDir = os.path.join(outputRootDir,'Pred3D')
    predictInput = os.path.join(ThreeDImageDir, 'FileList.txt')
    groundTruthInput = os.path.join(inputRootDir, 'TestData',GTFILE)

    predictOutput = os.path.join(outputRootDir, 'Pred3D_binaryLabel')
    if not os.path.exists(predictOutput):
        os.mkdir(predictOutput)
    
    dicorestat = CTP.CompareThePreandTruth(predictInput, groundTruthInput)
    dicorestat.readPredictImagetoList()
    dicorestat.readgroundTruthtoList()
    labelList = dicorestat.predictModification(predictOutput, preThreshold)
    WriteListtoFile(labelList, predictOutput + '/FileList.txt')
    dicorestat.diceScoreStatistics()
    
# evaluate the segmentation after filting with dice score 
def labelFiltering(inputRootDir,outputRootDir):
    predictOutput = os.path.join(outputRootDir, 'Pred3DMod')
    filepathLabel = os.path.join(predictOutput, 'FileList.txt')
    inputroot = os.path.join(inputRootDir, 'TestData')
    outputroot = os.path.join(outputRootDir, 'Pred3D_FFilter')
    
    postProcessing(filepathLabel, inputroot, outputroot)

    predictInput = os.path.join(outputroot, 'FileList.txt')       
    groundTruthInput = os.path.join(inputRootDir, 'TestData', GTFILE)    
    predictOutput = os.path.join(outputRootDir, 'Pred3DMod_FFilter')
    if not os.path.exists(predictOutput):
        os.mkdir(predictOutput)
    dicorestat = CTP.CompareThePreandTruth(predictInput, groundTruthInput)
    dicorestat.readPredictImagetoList()
    dicorestat.readgroundTruthtoList()
    dicorestat.predictModification(predictOutput, preThreshold)    
    dicorestat.diceScoreStatistics()      

def Pred3Ddata():
    predictDir = '../3D/'
    predictInput = os.path.join(predictDir, 'FileList.txt')
    groundTruthInput = os.path.join(inputRootDir, 'TestData', GTFILE)
    dicorestat = CTP.CompareThePreandTruth(predictInput, groundTruthInput)
    dicorestat.readPredictImagetoList()
    dicorestat.readgroundTruthtoList()
    dicorestat.diceScoreStatistics()  
    

def showlosscurve(tempStore):
    loss = np.load(os.path.join(tempStore,'loss.npy'))
    val_loss = np.load(os.path.join(tempStore,'val_loss.npy'))
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    plt.show()

def main():
    parser = argparse.ArgumentParser(description = "BATNet command line tool")
    parser.add_argument("--project-folder", type=str, help = "project folder to save the output data.")
    parser.add_argument("--reference-image-path", type=str, help = "path of a reference image to set origin, spacing and direction of output images")
    args = parser.parse_args()
    
    inputRootDir = args.project_folder
    outputRootDir = os.path.join(args.project_folder, config['dataDim'])
    refImage = args.reference_image_path
    tempStore = os.path.join(args.project_folder, config['dataDim'], 'tempData')

    # transform array data to images
    Test3DDataCollecting(inputRootDir,outputRootDir, refImage, sliceNum, img_rows, img_cols)
    # show trainging loss 
    showlosscurve(tempStore)
    # employing filter to further improve the predicted label
    diceComputing(inputRootDir,outputRootDir)


if __name__ == '__main__':
    main()