# -*- encoding: utf-8 -*-
'''
@File    :   semi_mask_target.py
@Time    :   2021/01/17 17:32:35
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2021
@Desc    :   get mask taget by unlabled image, used in semi-supervised learning framework
'''

import numpy as np
import torch
from torch.nn.modules.utils import _pair


def semi_mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg, offsets_list, only_footprint_flag_list):
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
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(semi_mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list, offsets_list, only_footprint_flag_list)
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = torch.cat(mask_targets)
    return mask_targets


def semi_mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg, offsets, only_footprint_flag):
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
    device = pos_proposals.device
    mask_size = _pair(cfg.mask_size)
    num_pos = pos_proposals.size(0)
    only_footprint_flag = only_footprint_flag.cpu().numpy()[0]
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw = gt_masks.height, gt_masks.width
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        # only annotate the footprint, gt mask if roof mask, there is no need to translate it
        if only_footprint_flag:
            gt_masks = gt_masks.translation(offsets, pos_assigned_gt_inds)

        mask_targets = gt_masks.crop_and_resize(
            proposals_np, mask_size, device=device,
            inds=pos_assigned_gt_inds).to_ndarray()

        mask_targets = torch.from_numpy(mask_targets).float().to(device)
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)

    return mask_targets
