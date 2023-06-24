# BONAI

This is the official code for the [BONAI](https://arxiv.org/abs/2204.13637) (TPAMI 2022). BONAI (**B**uildings in **O**ff-**N**adir **A**erial **I**mages) is a dataset for building footprint extraction (BFE) in off-nadir aerial images.

[[Paper]](https://arxiv.org/abs/2204.13637) [[Dataset]](https://drive.google.com/drive/folders/171PPLyEoIa67ZCuO8GSbnRJWreO-K0ac?usp=sharing)

<div align="center">
  <img src="resources/samples-jpg.jpg" width="600"/>
</div>

## Description

BONAI contains 268,958 building instances across 3,300 aerial images with fully annotated instance-level roof and footprint for each building as well as the corresponding offset vector. Compared to BONAI, existing BFE datasets only annotate building footprints.

The images of BONAI are taken from six representative cities of China, i.e., Shanghai, Beijing, Harbin, Jinan, Chengdu, and Xi'an, the detailed number of images and object instances per image set and city are reported in the below table.

<div align="center">
  <img src="resources/dataset-details.png" width="400"/>
</div>

## Download

You can download the dataset on [Google Driver](https://drive.google.com/drive/folders/171PPLyEoIa67ZCuO8GSbnRJWreO-K0ac?usp=sharing).

## Evaluation
Training, Validation and Testing sets are publicly available. The evaluation code has been updated in [bonai_evaluation.py](tools/bonai/bonai_evaluation.py). You can evaluate the model by:

```
python tools/bonai/bonai_evaluation.py --version bc_v100.02.08 --model bc_v100.02.08_offset_rcnn_r50_2x_public_20201028_rotate_offset_4_angles_without_image_rotation --city shanghai_xian_public
```

Note: Installing the [bstool](https://github.com/jwwangchn/bstool) code library is required to run the evaluation code.

## LOFT & FOA

The codes of LOFT and FOA is now publicly available. You can refer to [MMDetection](https://github.com/open-mmlab/mmdetection) to install and run this project.

## Contact

This repo is currently maintained by Jinwang Wang (jwwangchn@whu.edu.cn).

## Citing

If you use BONAI dataset, codebase or models in your research, please consider cite.

```
@article{wang2022bonai,
  author={Wang, Jinwang and Meng, Lingxuan and Li, Weijia and Yang, Wen and Yu, Lei and Xia, Gui-Song},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Learning to Extract Building Footprints from Off-Nadir Aerial Images}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2022.3162583}}
```

