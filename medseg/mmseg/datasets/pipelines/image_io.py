import cv2
import os
import os.path as osp
import numpy as np

def load_multislice_gray_png(imname, num_slice=3, zflip=False):
    # Load single channel image for general lung detection
    def get_slice_name(img_path, delta=0):
        if delta == 0:
            return img_path
        delta = int(delta)
        name_slice_list = img_path.split(os.sep)
        slice_idx = int(name_slice_list[-1][:-4])
        img_name = '%03d.png' % (slice_idx + delta)
        full_path = os.path.join('./', *name_slice_list[:-1], img_name)

        # if the slice is not in the dataset, use its neighboring slice
        while not os.path.exists(full_path):
            #print('file not found:', full_path)
            delta -= np.sign(delta)
            img_name = '%03d.png' % (slice_idx + delta)
            full_path = os.path.join('./', *name_slice_list[:-1], img_name)
            if delta == 0:
                break
        return full_path

    def _load_data(img_name, delta=0):
        img_name = get_slice_name(img_name, delta)
        if img_name not in data_cache.keys():
            data_cache[img_name] = cv2.imread(img_name, 0)
            if data_cache[img_name] is None:
                print('file reading error:', img_name, os.path.exists(img_name))
                assert not data_cache[img_name] == None
        return data_cache[img_name]


    def _load_multi_data(im_cur, imname, num_slice, zfilp=False):
        ims = [im_cur]
        # find neighboring slices of im_cure
        rel_pos = 1
        sequence_flag = False if (zflip and np.random.rand() > 0.5) else True
        if sequence_flag:
            for p in range((num_slice-1)//2):
                im_prev = _load_data(imname, - rel_pos * (p + 1))
                im_next = _load_data(imname, rel_pos * (p + 1))
                ims = [im_prev] + ims + [im_next]
            #when num_slice is even number,got len(ims) with num_slice-1. Add 1 slice.
            if num_slice%2 == 0:
                im_next = _load_data(imname, rel_pos * (p + 2))
                ims = ims + [im_next]
        else:
            for p in range((num_slice-1)//2):
                im_next = _load_data(imname, - rel_pos * (p + 1))
                im_prev = _load_data(imname, rel_pos * (p + 1))
                ims = [im_prev] + ims + [im_next]
            #when num_slice is even number,got len(ims) with num_slice-1. Add 1 slice.
            if num_slice%2 == 0:
                im_prev = _load_data(imname, rel_pos * (p + 2))
                ims = [im_prev] + ims
        return ims

    data_cache = {}
    im_cur = cv2.imread(imname, 0)
    ims = _load_multi_data(im_cur, imname, num_slice, zflip)
    #ims = [im.astype(float) for im in ims]
    # Support MIP MinIP as auxiliary channel.
    im = cv2.merge(ims)

    return im

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def load_data(image_tensor, center_idx, delta):
    cur_idx = int(center_idx + delta)
    depth,_,_ = image_tensor.shape
    cur_idx = clamp(cur_idx, 0, depth-1)
    return image_tensor[cur_idx, :, :]

