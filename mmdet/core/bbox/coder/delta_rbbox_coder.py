# -*- encoding: utf-8 -*-
'''
@File    :   delta_rbbox_coder.py
@Time    :   2021/01/17 17:31:03
@Author  :   Jinwang Wang
@Version :   1.0
@Contact :   jwwangchn@163.com
@License :   (C)Copyright 2017-2021
@Desc    :   rbbox encoder
'''

import numpy as np
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaRBBoxCoder(BaseBBoxCoder):
    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 encode_method='thetaobb'):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.encode_method = encode_method

    def encode(self, rbboxes, gt_rbboxes):
        assert rbboxes.size(0) == gt_rbboxes.size(0)
        if self.encode_method == 'thetaobb':
            assert rbboxes.size(-1) == gt_rbboxes.size(-1) == 5
            encoded_rbboxes = thetaobb2delta(rbboxes, gt_rbboxes, self.means, self.stds)
        else:
            raise(RuntimeError('do not support the encode mthod: {}'.format(self.encode_method)))
        
        return encoded_rbboxes

    def decode(self,
               rbboxes,
               pred_rbboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        assert pred_rbboxes.size(0) == rbboxes.size(0)
        if self.encode_method == 'thetaobb':
            decoded_rbboxes = delta2thetaobb(rbboxes, pred_rbboxes, self.means, self.stds,
                                        max_shape, wh_ratio_clip)
        else:
            raise(RuntimeError('do not support the encode mthod: {}'.format(self.encode_method)))

        return decoded_rbboxes

def thetaobb2delta(proposals, gt, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.)):
    # proposals: (x1, y1, x2, y2)
    # gt: (cx, cy, w, h, theta)
    assert proposals.size(0) == gt.size(0)

    proposals = proposals.float()
    gt = gt.float()

    px = proposals[..., 0]
    py = proposals[..., 1]
    pw = proposals[..., 2]
    ph = proposals[..., 3]
    pa = proposals[..., 4]

    gx = gt[..., 0]
    gy = gt[..., 1]
    gw = gt[..., 2]
    gh = gt[..., 3]
    ga = gt[..., 4]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    da = (ga - pa) * np.pi / 180

    deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas

def delta2thetaobb(rois,
                   deltas,
                   means=[0., 0., 0., 0., 0.],
                   stds=[1., 1., 1., 1., 1.],
                   max_shape=None,
                   wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    da = denorm_deltas[:, 4::5] * 180.0 / np.pi

    max_ratio = np.abs(np.log(wh_ratio_clip))

    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    px = (rois[:, 0]).unsqueeze(1).expand_as(dx)
    py = (rois[:, 1]).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2]).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3]).unsqueeze(1).expand_as(dh)
    pa = (rois[:, 4]).unsqueeze(1).expand_as(da)
    
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    ga = da + pa

    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[0])
        gy = gy.clamp(min=0, max=max_shape[1])
        gw = gw.clamp(min=0, max=max_shape[0])
        gh = gh.clamp(min=0, max=max_shape[1])
    thetaobbs = torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)
    return thetaobbs