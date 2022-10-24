import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import scipy
import random
from nibabel.affines import apply_affine
np.set_printoptions(precision=3, suppress=True)

#%%
# random zoom
def random_zoom(matrix, min_percentage=0.7, max_percentage=1.2, seed=0):
    np.random.seed(seed)
    z = np.random.sample() * (max_percentage-min_percentage) + min_percentage
    zoom_matrix = np.array([[z, 0, 0, 0],
                           [0, z, 0, 0],
                           [0, 0, z, 0],
                           [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, zoom_matrix)

# 3D Medical image rotation
def random_rotate3D(img_numpy, min_angle, max_angle, seed=0):
   """
   Returns a random rotated array in the same shape
   :param img_numpy: 3D numpy array
   :param min_angle: in degrees
   :param max_angle: in degrees
   """
   assert img_numpy.ndim == 3, "provide a 3d numpy array"
   assert min_angle < max_angle, "min should be less than max val"
   assert min_angle > -360 or max_angle < 360
   all_axes = [(1, 0), (1, 2), (0, 2)]
   np.random.seed(seed)
   angle = np.random.randint(low=min_angle, high=max_angle+1)
   axes_random_id = np.random.randint(low=0, high=len(all_axes))
   axes = all_axes[axes_random_id]
   return scipy.ndimage.rotate(img_numpy, angle, axes=axes)

# 3D medical image flip
def random_flip(img, label=None, seed=0):
   axes = [0, 1, 2]
   np.random.seed(seed)
   rand = np.random.randint(0, 3)
   img = flip_axis(img, axes[rand])
   img = np.squeeze(img)

   if label is None:
    return img
   else:
    label = flip_axis(label, axes[rand])
    label = np.squeeze(label)
   return img, label

def flip_axis(x, axis):
   x = np.asarray(x).swapaxes(axis, 0)
   x = x[::-1, ...]
   x = x.swapaxes(0, axis)
   return x


def crop_3d_volume(img_tensor, crop_dim, crop_size):
    assert img_tensor.ndim == 3, '3d tensor must be provided'
    full_dim1, full_dim2, full_dim3 = img_tensor.shape
    dim1, dim2, dim3 = crop_size
    
    if crop_dim:
        slices_crop, w_crop, h_crop = crop_dim
    else:
        slices_crop = int((full_dim1-dim1)/2)
        w_crop = int((full_dim2-dim2)/2)
        h_crop = int((full_dim3-dim3)/2)  

    # check if crop size matches image dimensions
    if full_dim1 == dim1:
        img_tensor = img_tensor[:, w_crop:w_crop + dim2,
                    h_crop:h_crop + dim3]
    elif full_dim2 == dim2:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, :,
                    h_crop:h_crop + dim3]
    elif full_dim3 == dim3:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2, :]
    # standard crop
    else:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2,
                    h_crop:h_crop + dim3]
    return img_tensor



def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


def augmentation_steps(image, label):
    seed = np.random.randint(1000)
    crop_size = np.shape(label)
    for i in range(len(image)):
        current_modality = image[i]
        current_modality = random_rotate3D(current_modality, min_angle=-5, max_angle=5, seed=seed)
        current_modality = crop_3d_volume(current_modality, crop_dim=None, crop_size=crop_size)
        current_modality = random_zoom(current_modality, min_percentage=0.7, max_percentage=1.2, seed=seed)
        current_modality = random_flip(current_modality, label=None, seed=seed)
        
        if i < 4:
            current_modality = augment_gaussian_noise(current_modality, noise_variance=(0, 0.1))
        image[i] = current_modality
    
    # label
    label = random_rotate3D(label, min_angle=-5, max_angle=5, seed=seed)
    label = crop_3d_volume(label, crop_dim=None, crop_size=crop_size)
    label = random_zoom(label, min_percentage=0.7, max_percentage=1.2, seed=seed)
    label = random_flip(label, label=None, seed=seed)

    return image, label



        
        
