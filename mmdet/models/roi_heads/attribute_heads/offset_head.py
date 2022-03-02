# -*- encoding: utf-8 -*-
'''
@File    :   offset_head.py
@Time    :   2021/01/17 20:42:55
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2021
@Desc    :   Core codes of offset head
'''

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init
from torch.nn.modules.utils import _pair

from mmdet.core import (build_bbox_coder, force_fp32, multi_apply)
from mmdet.models.builder import HEADS, build_loss
from mmcv.ops import Conv2d


@HEADS.register_module
class OffsetHead(nn.Module):
    def __init__(self,
                 roi_feat_size=7,
                 in_channels=256,
                 num_convs=4,
                 num_fcs=2,
                 reg_num=2,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 offset_coordinate='rectangle',
                 offset_coder=dict(
                    type='DeltaXYOffsetCoder',
                    target_means=[0.0, 0.0],
                    target_stds=[0.5, 0.5]),
                 reg_decoded_offset=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_offset=dict(type='MSELoss', loss_weight=1.0)):
        super(OffsetHead, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.offset_coordinate = offset_coordinate
        self.reg_decoded_offset = reg_decoded_offset
        self.reg_num = reg_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.offset_coder = build_bbox_coder(offset_coder)
        self.loss_offset = build_loss(loss_offset)

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_channels = (self.in_channels if i == 0 else self.conv_out_channels)
            self.convs.append(
                Conv2d(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    padding=1))
    
        roi_feat_size = _pair(roi_feat_size)
        roi_feat_area = roi_feat_size[0] * roi_feat_size[1]
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = (
                self.conv_out_channels *
                roi_feat_area if i == 0 else self.fc_out_channels)
            self.fcs.append(nn.Linear(in_channels, self.fc_out_channels))

        self.fc_offset = nn.Linear(self.fc_out_channels, self.reg_num)
        self.relu = nn.ReLU()
        self.loss_offset = build_loss(loss_offset)

    def init_weights(self):
        for conv in self.convs:
            kaiming_init(conv)
        for fc in self.fcs:
            kaiming_init(
                fc,
                a=1,
                mode='fan_in',
                nonlinearity='leaky_relu',
                distribution='uniform')
        normal_init(self.fc_offset, std=0.01)

    def forward(self, x):
        # self.vis_featuremap = x.clone()
        if x.size(0) == 0:
            return x.new_empty(x.size(0), 2)
        for conv in self.convs:
            x = self.relu(conv(x))
        
        self.vis_featuremap = x.clone()
        
        x = x.view(x.size(0), -1)
        # self.vis_featuremap = x.clone()
        for fc in self.fcs:
            x = self.relu(fc(x))
        offset = self.fc_offset(x)
        
        return offset

    @force_fp32(apply_to=('offset_pred', ))
    def loss(self, offset_pred, offset_targets):
        if offset_pred.size(0) == 0:
            loss_offset = offset_pred.sum() * 0
        else:
            loss_offset = self.loss_offset(offset_pred,
                                        offset_targets)
        return dict(loss_offset=loss_offset)

    def _offset_target_single(self,
                              pos_proposals, 
                              pos_assigned_gt_inds, 
                              gt_offsets, 
                              cfg):
        device = pos_proposals.device
        num_pos = pos_proposals.size(0)
        offset_targets = pos_proposals.new_zeros(pos_proposals.size(0), 2)

        pos_gt_offsets = []
        
        if num_pos > 0:
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            for i in range(num_pos):
                gt_offset = gt_offsets[pos_assigned_gt_inds[i]]
                pos_gt_offsets.append(gt_offset.tolist())

            pos_gt_offsets = np.array(pos_gt_offsets)
            pos_gt_offsets = torch.from_numpy(np.stack(pos_gt_offsets)).float().to(device)

            if not self.reg_decoded_offset:
                offset_targets = self.offset_coder.encode(pos_proposals, pos_gt_offsets)
            else:
                offset_targets = pos_gt_offsets
        else:
            offset_targets = pos_proposals.new_zeros((0, 2))

        return offset_targets, offset_targets

    def get_targets(self, 
                   sampling_results, 
                   gt_offsets, 
                   rcnn_train_cfg,
                   concat=True):
        """generate offset targets

        Args:
            sampling_results (torch.Tensor): sampling results
            gt_offsets (torch.Tensor): offset ground truth
            rcnn_train_cfg (dict): config of rcnn train
            concat (bool, optional): concat flag. Defaults to True.

        Returns:
            torch.Tensor: offset targets
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        offset_targets, _ = multi_apply(
            self._offset_target_single,
            pos_proposals,
            pos_assigned_gt_inds,
            gt_offsets,
            cfg=rcnn_train_cfg)
        
        if concat:
            offset_targets = torch.cat(offset_targets, 0)

        if self.reg_num == 2:
            return offset_targets
        elif self.reg_num == 3:
            length = offset_targets[:, 0]
            angle = offset_targets[:, 1]
            angle_cos = torch.cos(angle)
            angle_sin = torch.sin(angle)
            offset_targets = torch.stack([length, angle_cos, angle_sin], dim=-1)

            return offset_targets
        else:
            raise(RuntimeError("error reg_num value: ", self.reg_num))

    def get_offsets(self, 
                    offset_pred, 
                    det_bboxes,
                    scale_factor, 
                    rescale,
                    img_shape=[1024, 1024]):
        """get offsets in inference stage

        Args:
            offset_pred (torch.Tensor): predicted offset
            det_bboxes (torch.Tensor): detected bboxes
            scale_factor (int): scale factor
            rescale (int): rescale flag
            img_shape (list, optional): shape of image. Defaults to [1024, 1024].

        Returns:
            np.array: predicted offsets
        """
        if offset_pred is not None:
            if self.reg_num == 2:
                offsets = self.offset_coder.decode(det_bboxes, 
                                               offset_pred,
                                               max_shape=img_shape)
            elif self.reg_num == 3:
                length, angle_cos, angle_sin = offset_pred[:, 0], offset_pred[:, 1], offset_pred[:, 2]
                angle = torch.atan2(angle_sin, angle_cos)

                offset_pred = torch.stack([length, angle], dim=-1)

                offsets = self.offset_coder.decode(det_bboxes, 
                                               offset_pred,
                                               max_shape=img_shape)

            else:
                raise(RuntimeError("error reg_num value: ", self.reg_num))
        else:
            offsets = torch.zeros((det_bboxes.size()[0], self.reg_num))

        if isinstance(offsets, torch.Tensor):
            offsets = offsets.cpu().numpy()
        assert isinstance(offsets, np.ndarray)

        offsets = offsets.astype(np.float32)

        if self.offset_coordinate == 'rectangle':
            return offsets
        elif self.offset_coordinate == 'polar':
            length, angle = offsets[:, 0], offsets[:, 1]
            offset_x = length * np.cos(angle)
            offset_y = length * np.sin(angle)
            offsets = np.stack([offset_x, offset_y], axis=-1)
        else:
            raise(RuntimeError(f'do not support this coordinate: {self.offset_coordinate}'))

        return offsets

    def get_roof_footprint_bbox_offsets(self, 
                                        offset_pred, 
                                        det_bboxes,
                                        img_shape=[1024, 1024]):
        """decode the predicted offset

        Args:
            offset_pred (torch.Tensor): predicted offsets
            det_bboxes (torch.Tensor): predicted bboxes
            img_shape (list, optional): image shape. Defaults to [1024, 1024].

        Returns:
            np.array: decoded offsets
        """
        if offset_pred is not None:
            offsets = self.offset_coder.decode(det_bboxes, 
                                            offset_pred,
                                            max_shape=img_shape)
        else:
            offsets = torch.zeros((det_bboxes.size()[0], self.reg_num))

        return offsets