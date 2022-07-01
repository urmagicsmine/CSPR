import numpy as np
import nibabel as nb
import SimpleITK as sitk

def itk_load_nii(nii_path):
    image = sitk.ReadImage(nii_path)
    image_array = sitk.GetArrayFromImage(image)
    # Transform data into (h, w, d) format
    image_array = image_array.transpose(1, 2, 0)
    # Spacing in (h, w, d)
    return image_array, image.GetSpacing()

def save_nii(image_array, nii_path, image=None):
    target_image = sitk.GetImageFromArray(image_array)
    if image is not None:
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()
        target_image.SetSpacing(spacing)
        target_image.SetDirection(direction)
        target_image.SetOrigin(origin)
    sitk.WriteImage(target_image, nii_path)
    return nii_path

def load_image_nii(img_nii_fp, verbose=False):
    """load nii tensor and affine_matrix 
       Return tensor in (h, w, d) format, same as default mmcv output
    """
    nii_obj = nb.load(img_nii_fp)
    affine_matrix = nii_obj.affine
    x_col_sign = int(np.sign(-1 * affine_matrix[0,0]))
    y_row_sign = int(np.sign(-1 * affine_matrix[1,1]))
    if verbose: print('x y sign', x_col_sign, y_row_sign)
    if verbose: print('affine matrix\n', affine_matrix)
    image_3d = np.swapaxes(nii_obj.get_data(), 2, 0)
    image_3d = image_3d[:, ::y_row_sign, ::x_col_sign]
    spacing_list = nii_obj.header.get_zooms()[::-1]
    if verbose: print('pixel spacing', spacing_list)
    # Transform data into (h, w, d) format
    image_3d = image_3d.transpose(1, 2, 0)
    spacing_list = spacing_list[1:] + (spacing_list[0],)
    return image_3d, spacing_list

def get_affmat(img_nii_fp, verbose=False):
    """load nii tensor and affine_matrix 
       Return tensor in (h, w, d) format, same as default mmcv output
    """
    nii_obj = nb.load(img_nii_fp)
    affine_matrix = nii_obj.affine
    return affine_matrix


def adjust_ww_wl(image, ww = 400, wc = 40, is_uint8 = True):
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

def load_npz(npz_path):
    npz_data = np.load(npz_path)
    return npz_data['data'], list(npz_data['spacing_list'])

def array2nii(mask_3d, save_name, affine_matrix = None, nii_obj= None):
    """
    将输入图像存储为nii
    输入维度是z, r=y, c=x
    输出维度是x=c, y=r, z
    """
    mask_3d = np.swapaxes(mask_3d, 0, 2)
    # mask_3d = mask_3d[::-1,::-1,:] # be cautious to uncomment this line
    if nii_obj is None:
        if affine_matrix is None: affine_matrix = np.eye(4)
        nb_ojb = nb.Nifti1Image(mask_3d, affine_matrix)
    else:
        nb_ojb = nb.Nifti1Image(mask_3d, nii_obj.affine, nii_obj.header)

    nb.save(nb_ojb, save_name)
    return save_name
