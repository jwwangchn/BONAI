import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

from mmdet.apis import inference_detector
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

config_file = '../../configs/loft_foa/loft_foa_r50_fpn_2x_bonai.py'
#config_file = '../../configs/_base_/models/bonai_loft_foa_r50_fpn_basic.py'
config_file = '../../work_dirs/loft_foa_r50_fpn_2x_bonai/loft_foa_r50_fpn_2x_bonai.py'
checkpoint_file = '../../work_dirs/loft_foa_r50_fpn_2x_bonai/loft_foa_r50_fpn_2x_bonai_1-544d6bf6.pth'
cfg = Config.fromfile(config_file)

model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, checkpoint_file, map_location='cuda')
#model = fuse_module(model)
model.CLASSES = checkpoint['meta']['CLASSES']

model.with_vis_feat = False
# test a single image and show the results
img = './test.png'  # or img = mmcv.imread(img), which will only load it once

model.eval()

result = inference_detector(model, cfg, img)
#print("result = ", result)
#with torch.no_grad():
#    result = model(return_loss=False, rescale=True, **data)

# visualize the results in a new window
#model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='./result.png')