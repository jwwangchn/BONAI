# -*- encoding: utf-8 -*-
'''
@File    :   offset_field_target.py
@Time    :   2021/01/17 17:32:17
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2021
@Desc    :   get offset field target
'''

import numpy as np
import torch
from torch.nn.modules.utils import _pair
import mmcv


def offset_field_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_offset_fields,
                cfg):
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals.
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        list[Tensor]: Mask target of each image.
    """
    gt_offset_fields_list = []
    for idx in range(len(pos_proposals_list)):
        gt_offset_fields_list.append(gt_offset_fields[idx, ...])
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(offset_field_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_offset_fields_list, cfg_list)
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = torch.cat(mask_targets)
    return mask_targets


def offset_field_target_single(pos_proposals, pos_assigned_gt_inds, gt_offset_fields, cfg):
    """Compute mask target for each positive proposal in the image.

    Args:
        pos_proposals (Tensor): Positive proposals.
        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
            or Polygon.
        cfg (dict): Config dict that indicate the mask size.

    Returns:
        Tensor: Mask target of each positive proposals in the image.
    """
    gt_offset_fields = gt_offset_fields.squeeze().cpu().numpy()
    
    device = pos_proposals.device
    mask_size = _pair(cfg.mask_size)
    num_pos = pos_proposals.size(0)
    mask_targets = []
    
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw = gt_offset_fields.shape[0], gt_offset_fields.shape[1]
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        for i in range(num_pos):
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)

            gt_offset_fields[y1:y1 + h, x1:x1 + w, 0] = gt_offset_fields[y1:y1 + h, x1:x1 + w, 0] / w
            gt_offset_fields[y1:y1 + h, x1:x1 + w, 1] = gt_offset_fields[y1:y1 + h, x1:x1 + w, 1] / h
            target = mmcv.imresize(gt_offset_fields[y1:y1 + h, x1:x1 + w, :],
                                   mask_size[::-1])
            target = target.transpose(2, 0, 1)
            mask_targets.append(target)

        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(device)
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)

    return mask_targets
