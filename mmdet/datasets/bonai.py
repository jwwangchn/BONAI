import os
import os.path as osp
import tempfile
import pandas
import csv
import cv2

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import tqdm
import math

from shapely.validation import explain_validity
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon
from collections import defaultdict
from pycocotools.coco import COCO
import bstool

from mmcv.utils import print_log

from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module()
class BONAI(CocoDataset):
    CLASSES = ('building')
    def __init__(self,
                ann_file,
                pipeline,
                classes=None,
                data_root=None,
                img_prefix='',
                seg_prefix=None,
                edge_prefix=None,
                side_face_prefix=None,
                offset_field_prefix=None,
                proposal_file=None,
                test_mode=False,
                filter_empty_gt=True,
                gt_footprint_csv_file=None,
                bbox_type='roof',
                mask_type='roof',
                offset_coordinate='rectangle',
                resolution=0.6,
                ignore_buildings=True):
        super(BONAI, self).__init__(ann_file=ann_file,
                                                pipeline=pipeline,
                                                classes=classes,
                                                data_root=data_root,
                                                img_prefix=img_prefix,
                                                seg_prefix=seg_prefix,
                                                proposal_file=proposal_file,
                                                test_mode=test_mode,
                                                filter_empty_gt=filter_empty_gt)
        self.ann_file = ann_file
        self.bbox_type = bbox_type
        self.mask_type = mask_type
        self.offset_coordinate = offset_coordinate
        self.resolution = resolution
        self.ignore_buildings = ignore_buildings
        self.gt_footprint_csv_file = gt_footprint_csv_file

        self.edge_prefix = edge_prefix
        self.side_face_prefix = side_face_prefix
        self.offset_field_prefix = offset_field_prefix

        if self.data_root is not None:
            if not (self.edge_prefix is None or osp.isabs(self.edge_prefix)):
                self.edge_prefix = osp.join(self.data_root, self.edge_prefix)

        if self.data_root is not None:
            if not (self.side_face_prefix is None or osp.isabs(self.side_face_prefix)):
                self.side_face_prefix = osp.join(self.data_root, self.side_face_prefix)

        if self.data_root is not None:
            if not (self.offset_field_prefix is None or osp.isabs(self.offset_field_prefix)):
                self.offset_field_prefix = osp.join(self.data_root, self.offset_field_prefix)

        # print("This dataset has these keys: {}".format(list(self.get_properties(0))))

    def pre_pipeline(self, results):
        super(BONAI, self).pre_pipeline(results)
        results['edge_prefix'] = self.edge_prefix
        results['edge_fields'] = []

        results['side_face_prefix'] = self.side_face_prefix
        results['side_face_fields'] = []

        results['offset_field_prefix'] = self.offset_field_prefix
        results['offset_field_fields'] = []

    def get_properties(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)

        return ann_info[0].keys()

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            all_iscrowd = all([_['iscrowd'] for _ in ann_info])
            if self.filter_empty_gt and (self.img_ids[i] not in ids_with_ann
                                         or all_iscrowd):
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_roof_masks_ann = []
        gt_footprint_masks_ann = []
        gt_offsets = []
        gt_building_heights = []
        gt_angles = []
        gt_mean_angle = 0.0
        gt_roof_bboxes = []
        gt_footprint_bboxes = []
        gt_only_footprint_flag = 0

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            # bbox type may be roof, building and footprint, you need to set the value in config file
            if self.bbox_type == 'roof':
                x1, y1, w, h = ann['bbox']
            elif self.bbox_type == 'building':
                x1, y1, w, h = ann['building_bbox']
            elif self.bbox_type == 'footprint':
                x1, y1, w, h = ann['footprint_bbox']
            else:
                raise(TypeError(f"don't support bbox_type={self.bbox_type}"))

            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False) and self.ignore_buildings:
                gt_bboxes_ignore.append(bbox)
            else:
                if 'roof_bbox' in ann:
                    x1, y1, w, h = ann['roof_bbox']
                    gt_roof_bboxes.append([x1, y1, x1 + w, y1 + h])
                if 'footprint_bbox' in ann:
                    x1, y1, w, h = ann['footprint_bbox']
                    gt_footprint_bboxes.append([x1, y1, x1 + w, y1 + h])
                if 'only_footprint' in ann:
                    if ann['only_footprint'] == 1:
                        gt_only_footprint_flag = 1
                    else:
                        gt_only_footprint_flag = 0
                
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                # gt_only_footprint_flag=0: use roof as mask, gt_only_footprint_flag=0:use footprint as mask
                if gt_only_footprint_flag == 0:
                    if self.mask_type == 'roof':
                        gt_masks_ann.append(ann['segmentation'])
                    elif self.mask_type == 'footprint':
                        gt_masks_ann.append([ann['footprint_mask']])
                    else:
                        raise(TypeError(f"don't support mask_type={self.mask_type}"))
                else:
                    gt_masks_ann.append([ann['footprint_mask']])

                gt_roof_masks_ann.append(ann['segmentation'])
                gt_footprint_masks_ann.append([ann['footprint_mask']])

                # rectangle coordinate -> offset = (x, y), polar coordinate -> offset = (length, theta)
                if 'offset' in ann:
                    if self.offset_coordinate == "rectangle":
                        gt_offsets.append(ann['offset'])
                    elif self.offset_coordinate == 'polar':
                        offset_x, offset_y = ann['offset']
                        length = math.sqrt(offset_x ** 2 + offset_y ** 2)
                        angle = math.atan2(offset_y, offset_x)
                        gt_offsets.append([length, angle])
                    else:
                        raise(RuntimeError(f'do not support this coordinate: {self.offset_coordinate}'))
                else:
                    gt_offsets.append([0, 0])

                if 'building_height' in ann:
                    gt_building_heights.append(ann['building_height'])
                else:
                    gt_building_heights.append(0.0)

                if 'offset' in ann and 'building_height' in ann:
                    offset_x, offset_y = ann['offset']
                    height = ann['building_height']
                    angle = math.atan2(math.sqrt(offset_x ** 2 + offset_y ** 2) * self.resolution, height)

                    gt_angles.append(angle)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_roof_bboxes = np.array(gt_roof_bboxes, dtype=np.float32)
            gt_footprint_bboxes = np.array(gt_footprint_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_offsets = np.array(gt_offsets, dtype=np.float32)
            gt_building_heights = np.array(gt_building_heights, dtype=np.float32)
            gt_mean_angle = float(np.array(gt_angles, dtype=np.float32).mean())
            gt_only_footprint_flag = float(gt_only_footprint_flag)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_roof_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_footprint_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_offsets = np.zeros((0, 2), dtype=np.float32)
            gt_building_heights = np.zeros((0, 2), dtype=np.float32)
            gt_mean_angle = 0.0001
            gt_only_footprint_flag = 0

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')
        edge_map = img_info['filename'].replace('jpg', 'png')
        side_face_map = img_info['filename'].replace('jpg', 'png')
        offset_field = img_info['filename'].replace('png', 'npy')
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            roof_masks=gt_roof_masks_ann,
            footprint_masks=gt_footprint_masks_ann,
            seg_map=seg_map,
            offsets=gt_offsets,
            building_heights=gt_building_heights,
            angle=gt_mean_angle,
            edge_map=edge_map,
            side_face_map=side_face_map,
            roof_bboxes=gt_roof_bboxes,
            footprint_bboxes=gt_footprint_bboxes,
            offset_field=offset_field,
            only_footprint_flag=gt_only_footprint_flag)

        return ann

    def _segm2json(self, results):
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            if len(results[idx]) == 2:
                det, seg = results[idx]
            elif len(results[idx]) == 3:
                det, seg, offset = results[idx]
            elif len(results[idx]) == 4:
                det, seg, offset, building_height = results[idx]
            else:
                raise(RuntimeError("do not support the length of results: ", len(results[idx])))
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
       
        return bbox_json_results, segm_json_results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05),
                 detection_eval=True,
                 segmentation_eval=False,
                 model_name='',
                 cities=['xian'],
                 pred_csv_prefix=None,
                 gt_footprint_csv_file=None,
                 with_offset=False,
                 with_height=False,
                 save_merged_csv=False):
        """
        pred_csv_prefix: PATH/${model}_${city}
        """
        if detection_eval:
            coco_eval_results = super(BONAI, self).evaluate(results,
                metric=metric,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                classwise=classwise,
                proposal_nums=proposal_nums,
                iou_thrs=iou_thrs)

        if segmentation_eval:
            coco_eval_results = {}
            import bstool

            csv_info = 'splitted_on_training'
            summary_file = f'./data/buildchange/summary/{model_name}/{model_name}_xian_public_eval_summary_{csv_info}.csv'
            bstool.mkdir_or_exist(f'./data/buildchange/summary/{model_name}')

            anno_file = f'./data/buildchange/public/20201028/coco/annotations/buildchange_public_20201028_val_xian_fine.json'
            gt_roof_csv_file = './data/buildchange/public/20201028/xian_val_roof_crop1024_gt_minarea500.csv'
            gt_footprint_csv_file = './data/buildchange/public/20201028/xian_val_footprint_crop1024_gt_minarea500.csv'

            bstool.mkdir_or_exist(f'../mmdetv2-bc/results/buildchange/{model_name}')

            roof_csv_file = f'../mmdetv2-bc/results/buildchange/{model_name}/{model_name}_xian_public_roof_{csv_info}.csv'
            rootprint_csv_file = f'../mmdetv2-bc/results/buildchange/{model_name}/{model_name}_xian_public_footprint_{csv_info}.csv'

            evaluation = bstool.Evaluation(model=model_name,
                                            anno_file=self.ann_file,
                                            pkl_file=results,
                                            gt_roof_csv_file=gt_roof_csv_file,
                                            gt_footprint_csv_file=gt_footprint_csv_file,
                                            roof_csv_file=roof_csv_file,
                                            rootprint_csv_file=rootprint_csv_file,
                                            iou_threshold=0.1,
                                            score_threshold=0.4,
                                            with_offset=with_offset,
                                            show=False,
                                            save_merged_csv=False)
            # try:
            if evaluation.dump_result:
                segmentation_eval_results = evaluation.segmentation()
                meta_info = dict(summary_file=summary_file,
                                    model=model_name,
                                    anno_file=anno_file,
                                    gt_roof_csv_file=gt_roof_csv_file,
                                    gt_footprint_csv_file=gt_footprint_csv_file,
                                    vis_dir='')
                self.write_results2csv([segmentation_eval_results], meta_info)
                result_dict = {"Roof F1: ": segmentation_eval_results['roof']['F1_score'],
                                       "Roof Precition: ": segmentation_eval_results['roof']['Precision'],
                                       "Roof Recall: ": segmentation_eval_results['roof']['Recall'],
                                       "Footprint F1: ": segmentation_eval_results['footprint']['F1_score'],
                                       "Footprint Precition: ": segmentation_eval_results['footprint']['Precision'],
                                       "Footprint Recall: ": segmentation_eval_results['footprint']['Recall']}
            else:
                print('!!!!!!!!!!!!!!!!!!!!!! ALl the results of images are empty !!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # except:
                # print("Skip the segmentation evaluation")

        return coco_eval_results

    def write_results2csv(self, results, meta_info=None):
        print("meta_info: ", meta_info)
        segmentation_eval_results = results[0]
        with open(meta_info['summary_file'], 'w') as summary:
            csv_writer = csv.writer(summary, delimiter=',')
            csv_writer.writerow(['Meta Info'])
            csv_writer.writerow(['model', meta_info['model']])
            csv_writer.writerow(['anno_file', meta_info['anno_file']])
            csv_writer.writerow(['gt_roof_csv_file', meta_info['gt_roof_csv_file']])
            csv_writer.writerow(['gt_footprint_csv_file', meta_info['gt_footprint_csv_file']])
            csv_writer.writerow(['vis_dir', meta_info['vis_dir']])
            csv_writer.writerow([''])
            for mask_type in ['roof', 'footprint']:
                csv_writer.writerow([mask_type])
                csv_writer.writerow([segmentation_eval_results[mask_type]])
                csv_writer.writerow(['F1 Score', segmentation_eval_results[mask_type]['F1_score']])
                csv_writer.writerow(['Precision', segmentation_eval_results[mask_type]['Precision']])
                csv_writer.writerow(['Recall', segmentation_eval_results[mask_type]['Recall']])
                csv_writer.writerow(['True Positive', segmentation_eval_results[mask_type]['TP']])
                csv_writer.writerow(['False Positive', segmentation_eval_results[mask_type]['FP']])
                csv_writer.writerow(['False Negative', segmentation_eval_results[mask_type]['FN']])
                csv_writer.writerow([''])

            csv_writer.writerow([''])