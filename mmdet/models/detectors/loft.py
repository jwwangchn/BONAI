import numpy as np
import mmcv
import torch
import cv2
import math

from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class LOFT(TwoStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(LOFT, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.anchor_bbox_vis = [[287, 433, 441, 541]]
        self.with_vis_feat = True

    def show_result(self,
                    img,
                    result,
                    score_thr=0.8,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            if self.with_vis_feat:
                bbox_result, segm_result, offset, offset_features = result
            else:
                bbox_result, segm_result, offset = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        if isinstance(offset, tuple):
            offsets = offset[0]
        else:
            offsets = offset

        # rotate offset
        # offsets = self.offset_rotate(offsets, 0)

        bboxes = np.vstack(bbox_result)
        scores = bboxes[:, -1]
        bboxes = bboxes[:, 0:-1]

        w, h = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]
        area = w * h
        # valid_inds = np.argsort(area, axis=0)[::-1].squeeze()
        valid_inds = np.where(np.sqrt(area) > 50)[0]

        if segm_result is not None:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(scores > 0.4)[0][:]

            masks = []
            offset_results = []
            bbox_results = []
            offset_feats = []
            for i in inds:
                if i not in valid_inds:
                    continue
                mask = segms[i]
                offset = offsets[i]
                if self.with_vis_feat:
                    offset_feat = offset_features[i]
                else:
                    offset_feat = []
                bbox = bboxes[i]

                gray = np.array(mask * 255, dtype=np.uint8)

                contours = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                
                if contours != []:
                    cnt = max(contours, key = cv2.contourArea)
                    mask = np.array(cnt).reshape(1, -1).tolist()[0]
                else:
                    continue

                masks.append(mask)
                offset_results.append(offset)
                bbox_results.append(bbox)
                offset_feats.append(offset_feat)

    def offset_coordinate_transform(self, offset, transform_flag='xy2la'):
        """transform the coordinate of offsets

        Args:
            offset (list): list of offset
            transform_flag (str, optional): flag of transform. Defaults to 'xy2la'.

        Raises:
            NotImplementedError: [description]

        Returns:
            list: transformed offsets
        """
        if transform_flag == 'xy2la':
            offset_x, offset_y = offset
            length = math.sqrt(offset_x ** 2 + offset_y ** 2)
            angle = math.atan2(offset_y, offset_x)
            offset = [length, angle]
        elif transform_flag == 'la2xy':
            length, angle = offset
            offset_x = length * np.cos(angle)
            offset_y = length * np.sin(angle)
            offset = [offset_x, offset_y]
        else:
            raise NotImplementedError

        return offset

    def offset_rotate(self, offsets, rotate_angle):
        offsets = [self.offset_coordinate_transform(offset, transform_flag='xy2la') for offset in offsets]

        offsets = [[offset[0], offset[1] + rotate_angle * np.pi / 180.0] for offset in offsets]

        offsets = [self.offset_coordinate_transform(offset, transform_flag='la2xy') for offset in offsets]

        return np.array(offsets, dtype=np.float32)