import os
import os.path as osp
import numpy as np
import glob
import copy
import cv2
import random
import pickle
from scipy.ndimage import label, generate_binary_structure
from scipy import ndimage
import SimpleITK as sitk

class niiHandle():
    def __init__(self, nii_path=None):
        self.nii_path = nii_path
        self.itk_img = None
        self.img_array = None
        self.direction = None
        self.origin = None
        self.spacing = None

    def load_nii(self):
        self.itk_img = sitk.ReadImage(self.nii_path)
        self.direction = self.itk_img.GetDirection()
        self.origin = self.itk_img.GetOrigin()
        self.spacing = self.itk_img.GetSpacing()

    def save_nii(self, img_array, save_path, save_uint8=True):
        if save_uint8:
            img_sitk = sitk.GetImageFromArray(img_array.astype(np.uint8))
        else:
            img_sitk = sitk.GetImageFromArray(img_array)
        img_sitk.SetSpacing(self.spacing)
        img_sitk.SetOrigin(self.origin)
        img_sitk.SetDirection(self.direction)
        sitk.WriteImage(img_sitk, save_path)

    def set_path(self, nii_path):
        self.nii_path = nii_path

    def get_array(self):
        if self.itk_img is None:
            self.load_nii()
        self.img_array = sitk.GetArrayFromImage(self.itk_img)

    def transpose(self):
        direction = np.array(self.direction).reshape(3, 3)
        img_array = np.transpose(self.img_array, (2, 1, 0)) # [z, y, x] --> [x, y, z]
        y, x = np.where(direction != 0)
        img_array = np.transpose(img_array, x) # itk_img 原本的方向是[x, y, z]
        img_array = np.transpose(img_array, (2, 1, 0)) # [x, y, z] --> [z, y, x]
        for i in range(3):
            val = direction[y[i], x[i]]
            axes = x[i]
            if val < 0:
                img_array = np.flip(img_array, 2-axes)
        spacing = np.array(self.spacing)[np.array(x)][::-1]
        return img_array, spacing

    def get_spacing(self):
        if self.itk_img is None:
            self.load_nii()
        return self.itk_img.GetSpacing()

    def set_array(self, img_array):
        self.img_array = img_array

    def add_window(self, min_hu=-210, max_hu=300):
        self.img_array = ((self.img_array + 0.5 - min_hu) / (max_hu - min_hu) * 255)
        self.img_array[self.img_array > 255] = 255
        self.img_array[self.img_array < 0] = 0
        self.img_array = self.img_array.astype(np.uint8)

    def zoom(self, target_spacing, is_mask=False):
        if is_mask:
            pred_array = ndimage.zoom(self.img_array, (self.spacing[2]/target_spacing, 1, 1), order=0)
        else:
            pred_array = ndimage.zoom(self.img_array, (self.spacing[2]/target_spacing, 1, 1), order=3)
        return pred_array

    @staticmethod
    def adjust_ww_wl(image, ww = 510, wc = 45, is_uint8 = True):
        """
        adjust window width and window center to get proper input
        """
        min_hu = wc - (ww/2)
        max_hu = wc + (ww/2)
        new_image = np.clip(image, min_hu, max_hu)#np.copy(image)
        if is_uint8:
            new_image -= min_hu
            new_image = np.array(new_image / ww * 255., dtype = np.uint8)
        return new_image

    def show_img(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for index in range(self.img_array.shape[0]):
            png_name = os.path.join(save_folder, str(index).zfill(5)+".png")
            cv2.imwrite(png_name, self.img_array[index])

if __name__ == "__main__":
    root_folder = '/home/zhangshu/code/framework/medseg/data/LITS/mask_data/'
    nii_datas = glob.glob(root_folder + '*.nii')
    for nii_data in nii_datas:
        save_path = nii_data.replace('mask_data', 'liver_mask_data')
        print(save_path)
        nh = niiHandle()
        print(nii_data)
        nh.set_path(nii_data)
        nh.load_nii()
        nh.get_array()
        mask_array = nh.img_array
        print(np.sum(mask_array==2))
        mask_array[mask_array>0] = 1
        print(np.sum(mask_array==2))
        nh.save_nii(mask_array, save_path, save_uint8=True)


