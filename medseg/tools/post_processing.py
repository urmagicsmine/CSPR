import ast
from copy import deepcopy
from multiprocessing.pool import Pool
from nii_utils import niiHandle
import os
import os.path as osp

import numpy as np
from scipy.ndimage import label
import SimpleITK as sitk
import shutil


def load_remove_save(input_file: str, output_file: str, for_which_classes: list,
                     minimum_valid_object_size: dict = None):
    # Only objects larger than minimum_valid_object_size will be removed. Keys in minimum_valid_object_size must
    # match entries in for_which_classes
    handle = niiHandle()
    handle.set_path(input_file)
    handle.load_nii()
    handle.get_array()
    img_npy = handle.img_array
    img_in = handle.itk_img
    volume_per_voxel = float(np.prod(handle.spacing, dtype=np.float64))

    image, largest_removed, kept_size = remove_all_but_the_largest_connected_component(img_npy, for_which_classes,
                                                                                       volume_per_voxel,
                                                                                       minimum_valid_object_size)
    # print(input_file, "kept:", kept_size)
    handle.save_nii(image, output_file, save_uint8=True)
    return largest_removed, kept_size


def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size

def apply_postprocessing_to_folder(input_folder: str, output_folder: str, for_which_classes: list,
                                   min_valid_object_size:dict=None, num_processes=8):
    """
    applies removing of all but the largest connected component to all niftis in a folder
    :param min_valid_object_size:
    :param min_valid_object_size:
    :param input_folder:
    :param output_folder:
    :param for_which_classes:
    :param num_processes:
    :return:
    """
    if not osp.exists(output_folder):
        os.makedirs(output_folder)
    #maybe_mkdir_p(output_folder)
    #p = Pool(num_processes)
    nii_files = subfiles(input_folder, suffix=".nii", join=False)
    input_files = [osp.join(input_folder, i) for i in nii_files]
    out_files = [osp.join(output_folder, i) for i in nii_files]
    results = []
    for input_file, output_file in zip(input_files, out_files):
        load_remove_save(input_file, output_file, for_which_classes,
                          min_valid_object_size)

    #results = p.starmap_async(load_remove_save, zip(input_files, out_files, [for_which_classes] * len(input_files),
    #                                                [min_valid_object_size] * len(input_files)))
    #res = results.get()
    #p.close()
    #p.join()

def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def mumu_post_process(liver_seg):
    mask = label(liver_seg>0)[0]
    cls_id = 0
    cls_num = 0
    for _cls_id in np.unique(mask):
        if _cls_id==0:
            continue
        num = mask[mask==_cls_id].size
        if num>cls_num:
            cls_num = num 
            cls_id = _cls_id
    liver_seg[mask!=cls_id] = 0
    return liver_seg

if __name__ == "__main__":
   pass
