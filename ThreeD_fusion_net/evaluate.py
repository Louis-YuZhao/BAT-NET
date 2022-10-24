# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from skimage.metrics import hausdorff_distance

from config import config



#%%
rootdir =  os.path.abspath("../resultData/" )
acc_area = 0

def get_bat_segmentation(data):
    output = data == 1
    output.dtype = np.uint8
    return output

def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

def jaccard_index(truth, prediction):
    return 1 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction)- np.sum(truth * prediction))

def average_volume_difference(truth, prediction):
    return np.abs(np.sum(prediction,dtype=np.int32) - np.sum(truth, dtype=np.int32))/ np.sum(truth)

def recall(truth, prediction):
    return 1 * np.sum(truth * prediction)/ np.sum(truth)

def precision(truth, prediction):
    return  1 * np.sum(truth * prediction)/ np.sum(prediction)

def evaluate(prediction_dir):    
    header_choose = ("Bat Segmentation",)
    masking_functions_choose = (get_bat_segmentation,)
    headerlist = []
    masking_functions_list = []
    for i in range(len(header_choose)):
        if (i+1) in config["labels"]:
            headerlist.append(header_choose[i])
            masking_functions_list.append(masking_functions_choose[i])

    header = tuple(headerlist)
    masking_functions = tuple(masking_functions_list)
    rows = list()
    dice = list()
    case_folder_list = glob.glob(os.path.join(prediction_dir,"case*"))
    case_folder_list.sort()
    jaccard_list = []
    AVD_list = []
    precision_list = []
    recall_list = []
    hausdorff_list = []
    index_list = []
    for case_folder in case_folder_list:
        print(str(case_folder))
        truth_file = glob.glob(os.path.join(case_folder, "*truth.nii.gz"))[0]
        truth_image = sitk.ReadImage(truth_file)
        truth = sitk.GetArrayFromImage(truth_image)
        prediction_file = glob.glob(os.path.join(case_folder, "*prediction.nii.gz"))[0]
        prediction_image = sitk.ReadImage(prediction_file)
        prediction = sitk.GetArrayFromImage(prediction_image)    

        index = int(os.path.split(truth_file)[1].split('_')[1])

        # rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])
        dice_list = list()
        for func in masking_functions:
            if np.sum(func(truth)) > acc_area:
                dice_list.append(dice_coefficient(func(truth), func(prediction)))
            else:
                dice_list.append(np.int(0))
            print('truth:'+ str(np.sum(func(truth))))
            print('prediction:' + str(np.sum(func(prediction))))
            print('dice:' + str(dice_coefficient(func(truth), func(prediction))))
            print('jaccard:' + str(jaccard_index(func(truth), func(prediction))))
            print('AVD:' + str(average_volume_difference(func(truth), func(prediction))))
            print('recall:' + str(recall(func(truth), func(prediction))))
            print('precision:' + str(precision(func(truth), func(prediction))))
            print('hausdorff disctance:' + str(hausdorff_distance(func(truth).astype(bool), func(prediction).astype(bool))))
            jaccard_list.append(jaccard_index(func(truth), func(prediction)))
            AVD_list.append(average_volume_difference(func(truth), func(prediction)))
            recall_list.append(recall(func(truth), func(prediction)))
            precision_list.append(precision(func(truth), func(prediction)))
            hausdorff_list.append(hausdorff_distance(func(truth).astype(bool), func(prediction).astype(bool)))

        rows.append(dice_list)
        index_list.append(index)
        dice.append(dice_list[0])
    df = pd.DataFrame.from_records(rows, columns=header)
    df_1 = pd.DataFrame({'Dice': dice, 'Index': index_list})
    df_1.to_csv(os.path.join(prediction_dir,"test_scores.csv"), index=None)

    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        values = values[np.isnan(values) == False]
        scores[score] = values[values != np.int(0)]

    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.savefig(os.path.join(prediction_dir,"validation_scores_boxplot.png"))
    plt.show()
    plt.close()

    training_df = pd.read_csv(config["trainingLog"]).set_index('epoch')

    plt.plot(training_df['loss'].values, label='training loss')
    plt.plot(training_df['val_loss'].values, label='validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, len(training_df.index)))
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(prediction_dir, 'loss_graph.png'))
    plt.show()
    plt.close()
    
    print('Overall evaluation')
    print('dice:', 'mean:' + str(np.mean(rows)) , 'std:' + str(np.std(rows)), 'max:' + str(np.max(rows)), 'min:' + str(np.min(rows)))
    print('jaccard:', 'mean:' + str(np.mean(jaccard_list)) , 'std:' + str(np.std(jaccard_list)), 'max:' + str(np.max(jaccard_list)), 'min:' + str(np.min(jaccard_list)))
    print('AVD:', 'mean:' + str(np.mean(AVD_list)), 'std:' + str(np.std(AVD_list)), 'max:' + str(np.max(AVD_list)), 'min:' + str(np.min(AVD_list)))
    print('recall:','mean:' + str(np.mean(recall_list)) , 'std:' + str(np.std(recall_list)), 'max:' + str(np.max(recall_list)), 'min:' + str(np.min(recall_list)))
    print('precision:','mean:' + str(np.mean(precision_list)) , 'std:' + str(np.std(precision_list)), 'max:' + str(np.max(precision_list)), 'min:' + str(np.min(precision_list)))
    print('Hausdorff Distance:','mean:' + str(np.mean(hausdorff_list)) , 'std:' + str(np.std(hausdorff_list)), 'max:' + str(np.max(hausdorff_list)), 'min:' + str(np.min(hausdorff_list)))

    return np.mean(rows)
    
if __name__ == "__main__":
    prediction_dir = os.path.join(rootdir,'test_prediction')
    evaluate(prediction_dir)