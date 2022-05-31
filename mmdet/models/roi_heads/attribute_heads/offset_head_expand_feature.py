# -*- encoding: utf-8 -*-
'''
@File    :   offset_head_expand_feature.py
@Time    :   2021/01/17 20:18:09
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2021
@Desc    :   Main code for FOA module.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init
from torch.nn.modules.utils import _pair

from mmdet.core import (build_bbox_coder, force_fp32, multi_apply)
from mmdet.models.builder import HEADS, build_loss
from mmcv.ops import Conv2d
import math


@HEADS.register_module
class OffsetHeadExpandFeature(nn.Module):
    def __init__(self,
                 roi_feat_size=7,
                 in_channels=256,
                 num_convs=4,
                 num_fcs=2,
                 reg_num=2,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 expand_feature_num=4,
                 share_expand_fc=False,
                 rotations=[0, 90, 180, 270],
                 offset_coordinate='rectangle',
                 offset_coder=dict(
                    type='DeltaXYOffsetCoder',
                    target_means=[0.0, 0.0],
                    target_stds=[0.5, 0.5]),
                 reg_decoded_offset=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_offset=dict(type='MSELoss', loss_weight=1.0)):
        super(OffsetHeadExpandFeature, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.offset_coordinate = offset_coordinate
        self.reg_decoded_offset = reg_decoded_offset
        self.reg_num = reg_num
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # expand_feature_num is the branch numbers
        self.expand_feature_num = expand_feature_num
        self.share_expand_fc = share_expand_fc

        self.offset_coder = build_bbox_coder(offset_coder)
        self.loss_offset = build_loss(loss_offset)
        # the rotation angle of feature transformation
        self.rotations = rotations
        self.flips = ['h', 'v']

        # define the conv and fc operations
        self.expand_convs = nn.ModuleList()
        for _ in range(self.expand_feature_num):
            convs = nn.ModuleList()
            for i in range(num_convs):
                in_channels = (self.in_channels if i == 0 else self.conv_out_channels)
                convs.append(
                    Conv2d(
                        in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1))
            self.expand_convs.append(convs)
    
        roi_feat_size = _pair(roi_feat_size)
        roi_feat_area = roi_feat_size[0] * roi_feat_size[1]
        if self.share_expand_fc == False:
            self.expand_fcs = nn.ModuleList()
            for _ in range(self.expand_feature_num):
                fcs = nn.ModuleList()
                for i in range(num_fcs):
                    in_channels = (
                        self.conv_out_channels *
                        roi_feat_area if i == 0 else self.fc_out_channels)
                    fcs.append(nn.Linear(in_channels, self.fc_out_channels))
                self.expand_fcs.append(fcs)
            self.expand_fc_offsets = nn.ModuleList()
            for _ in range(self.expand_feature_num):
                fc_offset = nn.Linear(self.fc_out_channels, self.reg_num)
                self.expand_fc_offsets.append(fc_offset)
        else:
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
        for convs in self.expand_convs:
            for conv in convs:
                kaiming_init(conv)
        if self.share_expand_fc == False:
            for fcs in self.expand_fcs:
                for fc in fcs:
                    kaiming_init(
                        fc,
                        a=1,
                        mode='fan_in',
                        nonlinearity='leaky_relu',
                        distribution='uniform')
            for fc_offset in self.expand_fc_offsets:
                normal_init(fc_offset, std=0.01)
        else:
            for fc in self.fcs:
                kaiming_init(
                    fc,
                    a=1,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    distribution='uniform')
            normal_init(self.fc_offset, std=0.01)

    def forward(self, x):
        if x.size(0) == 0:
            return x.new_empty(x.size(0), 2 * self.expand_feature_num)
        input_feature = x.clone()
        offsets = []
        for idx in range(self.expand_feature_num):
            x = self.expand_feature(input_feature, idx)
            convs = self.expand_convs[idx]
            for conv in convs:
                x = self.relu(conv(x))

            x = x.view(x.size(0), -1)
            # share the fully connected parameters
            if self.share_expand_fc == False:
                fcs = self.expand_fcs[idx]
                for fc in fcs:
                    x = self.relu(fc(x))
                fc_offset = self.expand_fc_offsets[idx]
                offset = fc_offset(x)
            else:
                for fc in self.fcs:
                    x = self.relu(fc(x))
                offset = self.fc_offset(x)

            offsets.append(offset)

        offsets = torch.cat(offsets, 0)
        return offsets

    def expand_feature(self, feature, operation_idx):
        """rotate the feature by operation index

        Args:
            feature (torch.Tensor): input feature map
            operation_idx (int): operation index -> rotation angle

        Returns:
            torch.Tensor: rotated feature
        """
        if operation_idx < 4:
            # rotate feature map
            rotate_angle = self.rotations[operation_idx]
            theta = torch.zeros((feature.size()[0], 2, 3), requires_grad=False, device=feature.device)

            with torch.no_grad():
                # counterclockwise
                angle = rotate_angle * math.pi / 180.0
                
                theta[:, 0, 0] = torch.tensor(math.cos(angle), requires_grad=False, device=feature.device)
                theta[:, 0, 1] = torch.tensor(math.sin(-angle), requires_grad=False, device=feature.device)
                theta[:, 1, 0] = torch.tensor(math.sin(angle), requires_grad=False, device=feature.device)
                theta[:, 1, 1] = torch.tensor(math.cos(angle), requires_grad=False, device=feature.device)

            grid = F.affine_grid(theta, feature.size())
            transformed_feature = F.grid_sample(feature, grid).to(feature.device)

        elif operation_idx >= 4 and operation_idx < 8:
            # rotate and flip feature map
            raise NotImplementedError
        else:
            raise NotImplementedError

        return transformed_feature

    @force_fp32(apply_to=('offset_pred', ))
    def loss(self, offset_pred, offset_targets):
        if offset_pred.size(0) == 0:
            loss_offset = offset_pred.sum() * 0
        else:
            loss_offset = self.loss_offset(offset_pred,
                                        offset_targets)
        return dict(loss_offset=loss_offset)

    def offset_coordinate_transform(self, offset, transform_flag='xy2la'):
        """transform the coordinate of offsets

        Args:
            offset (list): list of offset
            transform_flag (str, optional): flag of transform. Defaults to 'xy2la'.

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
            offset_x = length * math.cos(angle)
            offset_y = length * math.sin(angle)
            offset = [offset_x, offset_y]
        else:
            raise NotImplementedError

        return offset

    def offset_rotate(self, offset, rotate_angle):
        """rotate the offset

        Args:
            offset (np.array): input offset
            rotate_angle (int): rotation angle

        Returns:
            np.array: rotated offset
        """
        offset = self.offset_coordinate_transform(offset, transform_flag='xy2la')
        # counterclockwise
        offset = [offset[0], offset[1] - rotate_angle * math.pi / 180.0]
        offset = self.offset_coordinate_transform(offset, transform_flag='la2xy')

        return offset

    def expand_gt_offset(self, gt_offset, operation_idx):
        """rotate the ground truth of offset

        Args:
            gt_offset (np.array): offset ground truth
            operation_idx (int): operation index

        Returns:
            np.array: rotated offset
        """
        if operation_idx < 4:
            # rotate feature map
            rotate_angle = self.rotations[operation_idx]
            transformed_offset = self.offset_rotate(gt_offset, rotate_angle)
        elif operation_idx >= 4 and operation_idx < 8:
            # rotate and flip feature map
            raise NotImplementedError
        else:
            raise NotImplementedError

        return transformed_offset

    def _offset_target_single(self,
                              pos_proposals, 
                              pos_assigned_gt_inds, 
                              gt_offsets, 
                              cfg,
                              operation_idx):
        # generate target of single item
        device = pos_proposals.device
        num_pos = pos_proposals.size(0)
        offset_targets = pos_proposals.new_zeros(pos_proposals.size(0), 2)

        pos_gt_offsets = []
        
        if num_pos > 0:
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            for i in range(num_pos):
                gt_offset = gt_offsets[pos_assigned_gt_inds[i]].tolist()
                gt_offset = self.expand_gt_offset(gt_offset, operation_idx=operation_idx)
                pos_gt_offsets.append(gt_offset)

            pos_gt_offsets = np.array(pos_gt_offsets)
            pos_gt_offsets = torch.from_numpy(np.stack(pos_gt_offsets)).float().to(device)

            if not self.reg_decoded_offset:
                if self.rotations[operation_idx] == 90 or self.rotations[operation_idx] == 270:
                    # if rotation angle is 90 or 270, the position of x and y need to be exchange
                    offset_targets = self.offset_coder.encode(pos_proposals, pos_gt_offsets[:, [1, 0]])
                    offset_targets = offset_targets[:, [1, 0]]
                else:
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
        """get the targets of offset in training stage

        Args:
            sampling_results (torch.Tensor): sampling results
            gt_offsets (torch.Tensor): offset ground truth
            rcnn_train_cfg (dict): rcnn training config
            concat (bool, optional): concat flag. Defaults to True.

        Returns:
            torch.Tensor: offset targets
        """
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        expand_offset_targets = []
        for idx in range(self.expand_feature_num):
            offset_targets, _ = multi_apply(
                self._offset_target_single,
                pos_proposals,
                pos_assigned_gt_inds,
                gt_offsets,
                cfg=rcnn_train_cfg,
                operation_idx=idx)
            
            if concat:
                offset_targets = torch.cat(offset_targets, 0)
            
            expand_offset_targets.append(offset_targets)
        
        expand_offset_targets = torch.cat(expand_offset_targets, 0)
        return expand_offset_targets

    def offset_fusion(self, offset_pred, model='max'):
        """Fuse the predicted offsets in inference stage

        Args:
            offset_pred (torch.Tensor): predicted offsets
            model (str, optional): fusion model. Defaults to 'max'. Max -> keep the max of offsets, Mean -> keep the mean value of offsets.

        Returns:
            np.array: fused offsets
        """
        split_offsets = offset_pred.split(int(offset_pred.shape[0]/self.expand_feature_num), dim=0)
        main_offsets = split_offsets[0]
        if model == 'mean':
            # Mean model for offset fusion
            offset_values = 0
            for idx in range(self.expand_feature_num):
                # 1. processing the rotation, rotation angle in (90, 270) -> switch the position of (x, y)
                if self.rotations[idx] == 90 or self.rotations[idx] == 270:
                    current_offsets = split_offsets[idx][:, [1, 0]]
                elif self.rotations[idx] == 0 or self.rotations[idx] == 180:
                    current_offsets = split_offsets[idx]
                else:
                    raise NotImplementedError(f"rotation angle: {self.rotations[idx]} (self.rotations = {self.rotations})")
                    
                offset_values += torch.abs(current_offsets)
            offset_values /= 1
        elif model == 'max':
            # Max model for offset fusion
            if self.expand_feature_num == 2 and self.rotations == [0, 180]:
                offset_value_x = torch.cat([split_offsets[0][:, 0].unsqueeze(dim=1), 
                                            split_offsets[1][:, 0].unsqueeze(dim=1)], dim=1)
                offset_value_y = torch.cat([split_offsets[0][:, 1].unsqueeze(dim=1), 
                                            split_offsets[1][:, 1].unsqueeze(dim=1)], dim=1)
            elif self.expand_feature_num == 2 and self.rotations == [0, 90]:
                offset_value_x = torch.cat([split_offsets[0][:, 0].unsqueeze(dim=1), 
                                            split_offsets[1][:, 1].unsqueeze(dim=1)], dim=1)
                offset_value_y = torch.cat([split_offsets[0][:, 1].unsqueeze(dim=1), 
                                            split_offsets[1][:, 0].unsqueeze(dim=1)], dim=1)
            elif self.expand_feature_num == 3 and self.rotations == [0, 90, 180]:
                offset_value_x = torch.cat([split_offsets[0][:, 0].unsqueeze(dim=1), 
                                            split_offsets[1][:, 1].unsqueeze(dim=1),
                                            split_offsets[2][:, 0].unsqueeze(dim=1)], dim=1)
                offset_value_y = torch.cat([split_offsets[0][:, 1].unsqueeze(dim=1), 
                                            split_offsets[1][:, 0].unsqueeze(dim=1),
                                            split_offsets[2][:, 1].unsqueeze(dim=1)], dim=1)
            elif self.expand_feature_num == 4:
                offset_value_x = torch.cat([split_offsets[0][:, 0].unsqueeze(dim=1), 
                                            split_offsets[1][:, 1].unsqueeze(dim=1), 
                                            split_offsets[2][:, 0].unsqueeze(dim=1), 
                                            split_offsets[3][:, 1].unsqueeze(dim=1)], dim=1)
                offset_value_y = torch.cat([split_offsets[0][:, 1].unsqueeze(dim=1), 
                                            split_offsets[1][:, 0].unsqueeze(dim=1), 
                                            split_offsets[2][:, 1].unsqueeze(dim=1), 
                                            split_offsets[3][:, 0].unsqueeze(dim=1)], dim=1)
            else:
                raise NotImplementedError

            offset_values = torch.cat([torch.max(torch.abs(offset_value_x), dim=1)[0].unsqueeze(dim=1), torch.max(torch.abs(offset_value_y), dim=1)[0].unsqueeze(dim=1)], dim=1)
        else:
            raise NotImplementedError
            
        offset_polarity = torch.zeros(main_offsets.size(), device=offset_pred.device)
        offset_polarity[main_offsets > 0] = 1
        offset_polarity[main_offsets <= 0] = -1

        fused_offsets = offset_values * offset_polarity

        return fused_offsets

    def get_offsets(self, 
                    offset_pred, 
                    det_bboxes,
                    scale_factor, 
                    rescale,
                    img_shape=[1024, 1024]):
        # generate offsets in inference stage
        if offset_pred is not None:
            # fuse the predicted offsets
            offset_pred = self.offset_fusion(offset_pred)
            # after offset fusion, the position of x and y is (x, y)
            offsets = self.offset_coder.decode(det_bboxes, 
                                               offset_pred,
                                               max_shape=img_shape)
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
        if offset_pred is not None:
            offsets = self.offset_coder.decode(det_bboxes, 
                                            offset_pred,
                                            max_shape=img_shape)
        else:
            offsets = torch.zeros((det_bboxes.size()[0], self.reg_num))

        return offsets
