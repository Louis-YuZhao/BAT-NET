import os
import argparse
import SimpleITK as sitk
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from skimage import morphology
from tqdm import tqdm
import glob


def make_mean_thresholding_mask(F_path, W_path):
    F_img = sitk.ReadImage(F_path)
    W_img = sitk.ReadImage(W_path)
    F_array = np.float32(sitk.GetArrayFromImage(F_img))
    W_array = np.float32(sitk.GetArrayFromImage(W_img))
    label_all = np.zeros(F_array.shape)

    for i in range(len(W_array)):
        
        label_i_1 = np.where(W_array[i] > W_array[i].mean(), 1, 0)
        label_i_2 = np.where(F_array[i] > F_array[i].mean(), 1, 0)
        label_all[i] = np.maximum(label_i_1, label_i_2)

    mask = np.int16(label_all)
    # sitk.WriteImage(sitk.GetImageFromArray(mask), 'demomask_threshold.nii.gz')

    return mask

def label_processing(label, FF_array, T2S_array):

    label_array = sitk.GetArrayFromImage(label).astype(int)
    
    FF_mask = np.where((FF_array > 35) & (FF_array < 110), 1, 0)
    T2S_mask = np.where(T2S_array > 10, 1, 0)

    label_processed = label_array * FF_mask * T2S_mask

    # smoothing
    label_processed = morphology.remove_small_objects(morphology.label(label_processed), 100)
    label_processed = np.where(label_processed > 0, 1, 0)
    label_img = sitk.GetImageFromArray(np.int16(label_processed))
    label_img.SetOrigin(label.GetOrigin())
    label_img.SetSpacing(label.GetSpacing())
    label_img.SetDirection(label.GetDirection())
    label_img = sitk.BinaryFillhole(label_img)
    label_img = sitk.BinaryDilate(label_img)
    label_img = sitk.BinaryErode(label_img)

    return label_img


def main():

    parser = argparse.ArgumentParser(description = "BATNet command line tool")
    parser.add_argument("--data-directory", type=str, help = "directory of data.")
    args = parser.parse_args()

    label_paths = glob.glob(os.path.join(args.data_directory, 'Labels/*.nii.gz'))
    label_paths.sort()

    for label_path in tqdm(label_paths):
        pathname, basename = os.path.split(label_path)
        subname = basename.split('_')[0] + '_' + basename.split('_')[1]
        main_dir = os.path.split(pathname)[0]
        F_dir = os.path.join(main_dir, 'F')
        W_dir = os.path.join(main_dir, 'W')
        FF_dir = os.path.join(main_dir, 'FF')
        T2S_dir = os.path.join(main_dir, 'T2S')
        F_path = os.path.join(F_dir, subname + '_F.nii.gz')
        W_path = os.path.join(W_dir, subname + '_W.nii.gz')
        FF_path = os.path.join(FF_dir, subname + '_FF.nii.gz')
        T2S_path = os.path.join(T2S_dir, subname + '_T2S.nii.gz')
        
        new_FF_dir = os.path.join(main_dir, 'FF_pre')
        if not os.path.isdir(new_FF_dir):
            os.makedirs(new_FF_dir)
        new_T2S_dir = os.path.join(main_dir, 'T2S_pre')
        if not os.path.isdir(new_T2S_dir):
            os.makedirs(new_T2S_dir)
        new_label_dir = os.path.join(main_dir, 'Label_processed')
        if not os.path.isdir(new_label_dir):
            os.makedirs(new_label_dir)
        
        
        # make thresholding mask
        label_all = make_mean_thresholding_mask(F_path, W_path)
        # process FF and T2S with thresholding mask
        FF_img = sitk.ReadImage(FF_path)
        T2S_img = sitk.ReadImage(T2S_path)
        FF_array = np.float32(sitk.GetArrayFromImage(FF_img))
        T2S_array = np.float32(sitk.GetArrayFromImage(T2S_img))
        FF_array = np.where(label_all>0, FF_array, 0)
        T2S_array = np.where(label_all>0, T2S_array, 0)

        new_FF_img = sitk.GetImageFromArray(FF_array)
        new_FF_img.SetOrigin(FF_img.GetOrigin())
        new_FF_img.SetSpacing(FF_img.GetSpacing())
        new_FF_img.SetDirection(FF_img.GetDirection())

        new_T2S_img = sitk.GetImageFromArray(T2S_array)
        new_T2S_img.SetOrigin(T2S_img.GetOrigin())
        new_T2S_img.SetSpacing(T2S_img.GetSpacing())
        new_T2S_img.SetDirection(T2S_img.GetDirection())


        sitk.WriteImage(new_FF_img, os.path.join(new_FF_dir, subname + '_FFpre.nii.gz'))
        sitk.WriteImage(new_T2S_img, os.path.join(new_T2S_dir, subname + '_T2Spre.nii.gz'))

        # label processing
        label = sitk.ReadImage(label_path)
        label_processed = label_processing(label, FF_array, T2S_array)
        sitk.WriteImage(label_processed, os.path.join(new_label_dir, subname + '_Label.nii.gz'))

    pass


if __name__ == "__main__":
    main()
