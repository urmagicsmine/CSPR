import os.path as osp
import numpy as np
import tempfile
import SimpleITK as sitk

import mmcv
from mmcv.utils import print_log
from .builder import DATASETS
from .custom import CustomDataset
from mmseg.core import mean_dice
from mmseg.utils import get_root_logger
from .pipelines.load_utils import load_image_nii, array2nii, get_affmat, save_nii, itk_load_nii
from .pipelines.image_io import load_multislice_gray_png
from tools.post_processing import mumu_post_process

@DATASETS.register_module()
class LiverDataset(CustomDataset):
    """Liver dataset, e.g. LITS.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 3 categories. ``reduce_zero_label`` is fixed to False. 

    Use 'split' to load img/ann infos, specifically with a txt of 'img_path mask_path'
    """

    CLASSES = ('background', 'liver', 'leison')

    PALETTE = [[120, 120, 120], [6, 230, 230], [224, 5, 255]]

    def __init__(self, num_slices=32, disp_per_case=False,  **kwargs):
        super(LiverDataset, self).__init__(
            img_dir=None,
            reduce_zero_label=False,
            img_suffix=None,
            seg_map_suffix=None,
            **kwargs)
        self.num_slices = num_slices
        self.disp_per_case = disp_per_case

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = img_info['ann']['seg_map']
            gt_seg_map = load_multislice_gray_png(seg_map, self.num_slices)
            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def evaluate(self, results, metric='mDice3D', logger=None, post_process=False, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU', 'mDice3D']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        gt_seg_maps = [gt_seg_map.transpose(2, 0, 1) for gt_seg_map in gt_seg_maps]
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        all_acc, acc, dice, case_dice = mean_dice(
            results, gt_seg_maps, num_classes, ignore_index=self.ignore_index, post_process=post_process)
        summary_str = '\n'
        if self.disp_per_case:
            for ind in range(len(results)):
                summary_str += '%-25s:' % osp.basename(self.img_infos[ind]['filename']) + str(case_dice[ind]) + '\n'
        summary_str += "MeanDice per case: " + str(np.mean(case_dice, axis=0)) + '\n'
        #summary_str = ''
        summary_str += 'per class results:\n'

        line_format = '{:<15} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'mDice3D', 'Acc')
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        remove_bg_item = lambda x: [x[ind] for ind in range(x.size) if class_names[ind] != 'background']
        acc, dice = list(map(remove_bg_item, [acc, dice]))
        class_names = [cls_name for cls_name in class_names if cls_name != 'background']

        for i in range(len(acc)):
            dice_str = '{:.2f}'.format(dice[i] * 100)
            acc_str = '{:.2f}'.format(acc[i] * 100)
            summary_str += line_format.format(class_names[i], dice_str, acc_str)
        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mDice3D', 'mAcc', 'aAcc')

        dice_str = '{:.2f}'.format(np.nanmean(dice) * 100)
        acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
        all_acc_str = '{:.2f}'.format(all_acc * 100)
        summary_str += line_format.format('global', dice_str, acc_str,
                                          all_acc_str)
        print_log(summary_str, logger)

        eval_results['mDice3D'] = np.nanmean(dice)
        eval_results['mAcc'] = np.nanmean(acc)
        eval_results['aAcc'] = all_acc

        return eval_results


@DATASETS.register_module()
class NiiLiverDataset(LiverDataset):
    """Liver dataset, e.g. LITS.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 3 categories. ``reduce_zero_label`` is fixed to False. 

    Use 'split' to load img/ann infos, specifically with a txt of 'img_path mask_path'
    """

    CLASSES = ('background', 'liver', 'leison')

    PALETTE = [[120, 120, 120], [6, 230, 230], [224, 5, 255]]

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = img_info['ann']['seg_map']
            gt_seg_map, _ = itk_load_nii(seg_map) 
            #gt_seg_map, _ = load_image_nii(seg_map)
            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def results2img(self, results, imgfile_prefix, to_label_id, post_process=False):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        if post_process:
            #print([type(result) for result in results])
            results = [mumu_post_process(result) for result in results if result is not None]
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            #affine_matrix = get_affmat(filename)
            basename = osp.splitext(osp.basename(filename))[0].split('-')[-1]

            nii_filename = osp.join(imgfile_prefix, 'test-segmentation-%s.nii' % basename)

            #affine_matrix = np.eye(4)
            #affine_matrix[0 , 0] = -1 # x, y, z? 
            #affine_matrix[2 , 2] = -1
            #x_col_sign = int(np.sign(-1 * affine_matrix[0,0]))
            #y_row_sign = int(np.sign(-1 * affine_matrix[1,1]))
            #nii_fp = array2nii(result[:,::y_row_sign,::x_col_sign].astype('uint8'),
            #                   nii_filename, affine_matrix= affine_matrix)
            ct_image = sitk.ReadImage(filename) 
            nii_fp = save_nii(result.astype('uint8'), nii_filename, image=ct_image)
            result_files.append(nii_fp)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=False, post_process=False):
        """Format the results into dir (standard format for LTIS evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            #tmp_dir = tempfile.TemporaryDirectory()
            #imgfile_prefix = tmp_dir.name
            imgfile_prefix = './submmit_results'
            tmp_dir='./submmit_results'
        else:
            tmp_dir = None
        print('\nsaveing nii to:', imgfile_prefix)
        result_files = self.results2img(results, imgfile_prefix, to_label_id, post_process)

        return result_files, tmp_dir

@DATASETS.register_module()
class NiiPancreasDataset(NiiLiverDataset):

    CLASSES = ('background', 'Pancreas')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

@DATASETS.register_module()
class NiiOLiverDataset(NiiLiverDataset):

    CLASSES = ('background', 'Liver')
    PALETTE = [[120, 120, 120], [6, 230, 230]]

@DATASETS.register_module()
class LiverRsDataset(CustomDataset):
    """Liver dataset, e.g. LITS. 
    Use slice image as input, this dataset only support training. 
    Randomly Sample a key slice when loading image/seg pairs.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 3 categories. ``reduce_zero_label`` is fixed to False. 

    Use 'split' to load img/ann infos, specifically with a txt of 'img_path mask_path'
    """

    CLASSES = ('background', 'liver', 'leison')
    PALETTE = [[120, 120, 120], [6, 230, 230], [224, 5, 255]]

    def __init__(self, num_slices=32, disp_per_case=False,  **kwargs):
        super(LiverRsDataset, self).__init__(
            img_dir=None,
            reduce_zero_label=False,
            img_suffix=None,
            seg_map_suffix=None,
            **kwargs)
        self.num_slices = num_slices
        self.disp_per_case = disp_per_case

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    if len(line.strip().split()) == 4:
                        # Added to support 'image_path seg_path'
                        img_name, seg_name, min_idx, max_idx = line.strip().split()
                        png_range = (int(min_idx), int(max_idx))
                        img_info = dict(filename=osp.join(self.data_root, img_name))
                        #if ann_dir is not None:
                        seg_map = seg_name
                        img_info['ann'] = dict(seg_map=osp.join(self.data_root, seg_map))
                        img_info['png_range'] = dict(png_range=png_range)
                        img_infos.append(img_info)
                    else: 
                        raise NotImplementedError
        else:
            raise NotImplementedError
        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        range_info = self.img_infos[idx]['png_range']
        results = dict(img_info=img_info, ann_info=ann_info, range_info=range_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        print("This dataset does not support testing")
        raise NotImplementedError

@DATASETS.register_module()
class OLiverRsDataset(LiverRsDataset):
    CLASSES = ('background', 'Liver')
    PALETTE = [[120, 120, 120], [6, 230, 230]]


@DATASETS.register_module()
class LidcDataset(LiverDataset):
    """Lidc dataset, e.g. LITS.
    """

    CLASSES = ('background', 'nodule')

    PALETTE = [[120, 120, 120], [6, 230, 230]]

    def __init__(self, crop_size=(48, 48, 48),  **kwargs):
        super(LidcDataset, self).__init__(
            **kwargs)
        self.crop_size = crop_size

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = img_info['ann']['seg_map']
            with np.load(seg_map) as npz:
                gt_seg_map = npz['answer1']
            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            shape = gt_seg_map.shape
            center = np.array(shape) // 2
            gt_seg_map = self.center_crop(center, gt_seg_map)
            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def center_crop(self, crop_center, img):
        y, x, z = crop_center
        h, w, d = self.crop_size
        cropped = img[y - h // 2:y + h // 2,
                      x - w // 2:x + w // 2,
                      z - d // 2:z + d // 2]
        return cropped

    def format_results(self, results, imgfile_prefix=None, to_label_id=False):
        """Format the results into dir (standard format for LTIS evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            #tmp_dir = tempfile.TemporaryDirectory()
            #imgfile_prefix = tmp_dir.name
            imgfile_prefix = './lidc_visual'
            tmp_dir='./lidc_visual'
        else:
            tmp_dir = None
        print('\nsaveing nii to:', imgfile_prefix)
        result_files = self.results2img(results, imgfile_prefix, to_label_id)

        return result_files, tmp_dir

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            with np.load(filename) as npz:
                img, gt_seg_map = npz['voxel'], npz['answer1']
            shape = gt_seg_map.shape
            center = np.array(shape) // 2
            gt_seg_map = self.center_crop(center, gt_seg_map)
            img = self.center_crop(center, img)
            #affine_matrix = get_affmat(filename)
            basename = osp.splitext(osp.basename(filename))[0]

            pred_nii_filename = osp.join(imgfile_prefix, '%s_pred.nii' % basename)
            gt_nii_filename = osp.join(imgfile_prefix, '%s_gt.nii' % basename)
            img_nii_filename = osp.join(imgfile_prefix, '%s_img.nii' % basename)

            pred_nii_fp = save_nii(result.astype('uint8'), pred_nii_filename, image=None)
            gt_nii_fp = save_nii(gt_seg_map.astype('uint8'), gt_nii_filename, image=None)
            img_nii_fp = save_nii(img, img_nii_filename, image=None)
            
            result_files.append(pred_nii_fp)
            prog_bar.update()

        return result_files
