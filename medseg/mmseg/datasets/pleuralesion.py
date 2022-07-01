import os.path as osp
import numpy as np
import tempfile
import SimpleITK as sitk
import cv2
import pickle
from PIL import Image
import torchvision.utils as vutils
import torchvision.transforms as T

import mmcv
from mmcv.utils import print_log
from .builder import DATASETS
from .custom import CustomDataset
from mmseg.core import mean_dice
from mmseg.utils import get_root_logger
from .pipelines.load_utils import load_image_nii, array2nii, get_affmat, save_nii, itk_load_nii
from .pipelines.image_io import load_multislice_gray_png
from tools.post_processing import mumu_post_process
import pdb 
from random import randint

@DATASETS.register_module()
class PleuraDataset(CustomDataset):
    """Pleura dataset, e.g. LITS.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 3 categories. ``reduce_zero_label`` is fixed to False. 

    Use 'split' to load img/ann infos, specifically with a txt of 'img_path mask_path'
    """

    CLASSES = ('background', 'pleuraeffusion', 'pneumothorax')

    PALETTE = [[120, 120, 120], [6, 230, 230], [224, 5, 255]]

    def __init__(self, num_slices=3, disp_per_case=False, disp_thresh=None,  **kwargs):
        super(PleuraDataset, self).__init__(
            img_dir=None,
            reduce_zero_label=False,
            img_suffix=None,
            seg_map_suffix=None,
            **kwargs)
        self.num_slices = num_slices
        self.disp_per_case = disp_per_case
        self.disp_thresh = disp_thresh # If not None, will only display those cases that are below this threshold.

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = img_info['ann']['seg_map']
            gt_seg_map = mmcv.imread(
                seg_map, flag='unchanged', backend='pillow')
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
        # gt_seg_maps = [gt_seg_map.transpose(2, 0, 1) for gt_seg_map in gt_seg_maps]
        
        # LJ 可视化分割结果
        # for idx in range(len(results)):
        #     cv2.imwrite('vis/gt' + str(idx) + '.png', gt_seg_maps[idx] * 100)
        #     cv2.imwrite('vis/pred' + str(idx) + '.png', results[idx] * 100)

        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        all_acc, acc, dice, case_dice = mean_dice(
            results, gt_seg_maps, num_classes, ignore_index=self.ignore_index, post_process=post_process)
        summary_str = '\n'
        if self.disp_per_case:
            if self.disp_thresh is None:
                for ind in range(len(results)):
                    summary_str += '%-25s:' % osp.basename(self.img_infos[ind]['filename']) + str(case_dice[ind]) + '\n'
            else:
                for ind in range(len(results)):
                    if case_dice[ind] <= self.disp_thresh:
                        summary_str += '%-25s:' % osp.basename(self.img_infos[ind]['filename']) + str(case_dice[ind]) + '\n'
        summary_str += "MeanDice per case: " + str(np.mean(case_dice, axis=0)) + '\n'
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

        def voc_colormap(N=256):
            bitget = lambda val, idx: ((val & (1 << idx)) != 0)

            cmap = np.zeros((N, 3), dtype=np.uint8)
            for i in range(N):
                r = g = b = 0
                c = i
                for j in range(8):
                    r |= (bitget(c, 0) << 7 - j)
                    g |= (bitget(c, 1) << 7 - j)
                    b |= (bitget(c, 2) << 7 - j)
                    c >>= 3

                cmap[i, :] = [r, g, b]
            return cmap

        # with open('./tools/voc_cmap.pkl', 'rb') as f:
        #     cmap = pickle.load(f)
        cmap = voc_colormap()
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            filename = self.img_infos[idx]['filename']
            basename = filename.replace('/', '_').split('.png')[0] #osp.splitext(osp.basename(filename))[0]
            img = Image.open(filename)
            img = img.convert('RGB')
            img_med = Image.open(filename.replace('image_links', 'image_links_med'))
            img_med = img_med.convert('RGB')
            # LJ 有些数据损坏无法读取 
            # try:
            #     img_med = Image.open(filename.replace('image_links', 'image_links_med'))
            #     img_med = img_med.convert('RGB')
            # except:
            #     continue
            gt_name = self.img_infos[idx]['ann']['seg_map']
            gt_img = Image.open(gt_name)
            gt_rgb = gt_img.convert('RGB')
            #cv2.imwrite(osp.join(imgfile_prefix, filename.replace('/', '_')), img)

            png_filename = osp.join(imgfile_prefix, f'{basename}_pred.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            output.putpalette(cmap)
            output_rgb = output.convert('RGB')  

            gt_blend = Image.blend(img, gt_rgb, alpha=0.5)
            # gt_img.putpalette(cmap)
            pred_blend = Image.blend(img, output_rgb, alpha=0.5)

            mini_batch = []
            mini_batch.append(T.functional.to_tensor(img_med))
            mini_batch.append(T.functional.to_tensor(img))
            mini_batch.append(T.functional.to_tensor(gt_blend)) 
            mini_batch.append(T.functional.to_tensor(pred_blend))    
            vutils.save_image(mini_batch, png_filename) 
            
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

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
            imgfile_prefix = './fcn_resnet18_unet_std_pretrained_aug_512x512_peseg_sgd_mview6_trainval_39k_bs8_lr1e4_fp16'
            tmp_dir='./pleura_visual'
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix, to_label_id)

        return result_files, tmp_dir
