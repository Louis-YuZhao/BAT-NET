# -*- coding: utf-8 -*-
"""
Post Processing after Unet.
"""
import os
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import utility.CompareThePreandtruth as CTP
from utility.LabelpostProcess import postProcessing
from config import config

GTFILE='Label.txt'
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

#%%
def Test3DDataCollecting(outputRootDir, tempStore, sliceNum, image_rows, image_cols):
    ThreeDImageDir = os.path.join(outputRootDir, 'Pred3D')
    if not os.path.exists(ThreeDImageDir):
        os.mkdir(ThreeDImageDir)    
    Reference=config['Image_Reference']
    VolumeDataToVolumes(ThreeDImageDir, tempStore, sliceNum, image_rows, image_cols)

def VolumeDataToVolumes(ThreeDImageDir, tempStore, sliceNum, image_rows, image_cols):

    imgs_label_test = np.load(os.path.join(tempStore,'imgs_label_test.npy'))
    dimZ, _, _, _ = np.shape(imgs_label_test)
    NumOfImage = int(dimZ/sliceNum)
    
    imgs_id_test = np.load(os.path.join(tempStore, 'imgs_id_test.npy'))

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
        threeDimage.SetOrigin(config['Image_Reference']['origin'])                               
        threeDimage.SetSpacing(config['Image_Reference']['spacing'])                                
        threeDimage.SetDirection(config['Image_Reference']['direction'])
        
        ThreeDImagePath = os.path.join(ThreeDImageDir, imgs_id_test[i]+'_pred'+'.nrrd')
        sitk.WriteImage(threeDimage, ThreeDImagePath)
        ImageList.append(ThreeDImagePath)

    WriteListtoFile(ImageList, ThreeDImageDir + '/FileList.txt')
    return ImageList

def WriteListtoFile(filelist, filename):
    with open(filename, 'w') as f:
        for i in filelist:
            f.write(i+'\n')
    return 1 

# evaluate the segmentation with dice score    
def diceComputing(inputRootDir, outputRootDir, threshold):        
    ThreeDImageDir = os.path.join(outputRootDir,'Pred3D')
    predictInput = os.path.join(ThreeDImageDir, 'FileList.txt')
    groundTruthInput = os.path.join(inputRootDir, 'TestData',GTFILE)

    predictOutput = os.path.join(outputRootDir, 'Pred3DMod')
    if not os.path.exists(predictOutput):
        os.mkdir(predictOutput)
    
    dicorestat = CTP.CompareThePreandTruth(predictInput, groundTruthInput)
    dicorestat.readPredictImagetoList()
    dicorestat.readgroundTruthtoList()
    labelList = dicorestat.predictModification(predictOutput, threshold)
    WriteListtoFile(labelList, predictOutput + '/FileList.txt')
    dicorestat.diceScoreStatistics()
    
# evaluate the segmentation after filting with dice score 
def labelFiltering(inputRootDir, outputRootDir, threshold):

    filepathLabel = os.path.join(outputRootDir, 'Pred3DMod', 'FileList.txt')
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
    dicorestat.predictModification(predictOutput, threshold)    
    dicorestat.diceScoreStatistics()      

def showlosscurve(tempStore):
    loss = np.load(os.path.join(tempStore,'loss.npy'))
    val_loss = np.load(os.path.join(tempStore,'val_loss.npy'))
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    # plt.show()    

def main():
   
    parser = argparse.ArgumentParser(description = "BATNet command line tool")
    parser.add_argument("--data_folder", type=str, default = '../', help = "Dataset Folder")
    parser.add_argument("--project_folder", type=str, help = "project folder to save the output data.")
    parser.add_argument("--threshold", type=float, default=0.5,  help = "classification threshold.")
    args = parser.parse_args()

    outputRootDir = os.path.join(args.project_folder, config['dataDim'])
    tempStore = os.path.join(args.project_folder, config['dataDim'], 'tempData')
    # transform array data to images
    Test3DDataCollecting(outputRootDir, tempStore, sliceNum, img_rows, img_cols)
    # show trainging loss 
    showlosscurve(tempStore)
    # employing filter to further improve the predicted label
    diceComputing(args.data_folder, outputRootDir, args.threshold)
    labelFiltering(args.data_folder, outputRootDir, args.threshold)

if __name__ == '__main__':
    main()