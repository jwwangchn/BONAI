# -*- encoding: utf-8 -*-
'''
@File    :   delta_polar_offset_coder.py
@Time    :   2021/01/17 17:30:31
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2021
@Desc    :   encode offset in polar coordinate
'''

import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaPolarOffsetCoder(BaseBBoxCoder):
    def __init__(self,
                 target_means=(0., 0.),
                 target_stds=(0.5, 0.5),
                 with_bbox=True):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.with_bbox = with_bbox

    def encode(self, bboxes, gt_offsets):
        assert bboxes.size(0) == gt_offsets.size(0)
        assert gt_offsets.size(-1) == 2
        encoded_offsets = offset2delta(bboxes, gt_offsets, self.means, self.stds, self.with_bbox)
        return encoded_offsets

    def decode(self,
               bboxes,
               pred_offsets,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        assert pred_offsets.size(0) == bboxes.size(0)
        decoded_offsets = delta2offset(bboxes, pred_offsets, self.means, self.stds,
                                    max_shape, wh_ratio_clip, self.with_bbox)

        return decoded_offsets

def offset2delta(proposals, gt, means=(0., 0.), stds=(0.5, 0.5), with_bbox=True):
    assert proposals.size()[0] == gt.size()[0]

    proposals = proposals.float()
    gt = gt.float()
    proposal_w = proposals[..., 2] - proposals[..., 0]
    proposal_h = proposals[..., 3] - proposals[..., 1]

    gt_length = gt[..., 0]
    gt_angle = gt[..., 1]

    proposal_length = torch.sqrt(proposal_w ** 2 + proposal_h ** 2)

    if with_bbox:
        delta_length = gt_length / proposal_length
    else:
        delta_length = gt_length
    delta_angle = gt_angle
    deltas = torch.stack([delta_length, delta_angle], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas

def delta2offset(rois,
                 deltas,
                 means=(0., 0.),
                 stds=(1., 1.),
                 max_shape=None,
                 wh_ratio_clip=16 / 1000,
                 with_bbox=True):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 2)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 2)
    denorm_deltas = deltas * stds + means
    delta_length = denorm_deltas[:, 0::2]
    delta_angle = denorm_deltas[:, 1::2]
    # Compute width/height of each roi
    proposal_w = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(delta_length)
    proposal_h = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(delta_angle)
    # Use network energy to shift the center of each roi
    proposal_length = torch.sqrt(proposal_w ** 2 + proposal_h ** 2)
    if with_bbox:
        gt_length = proposal_length * delta_length
    else:
        gt_length = delta_length
    gt_angle = delta_angle
    if max_shape is not None:
        gt_length = gt_length.clamp(min=-max_shape[1], max=max_shape[1])
        delta_angle = delta_angle.clamp(min=-max_shape[0], max=max_shape[0])
    bboxes = torch.stack([gt_length, gt_angle], dim=-1).view_as(deltas)
    return bboxes
