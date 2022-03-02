#!/usr/bin/env bash
# 输入模型名，使用 shanghai xian 公开数据集生成 pkl 检测文件
#------------------------------config-----------------------------------
config=${config}
epoch=${epoch:-latest}
city=${city:-shanghai_xian}

model_file=${config##*/}
model=${model_file%.*}

root=$(pwd)/results/bonai/${model}

mkdir -p ${root}

#------------------------------inference and eval-----------------------------------
echo "==== start evaluating, mode name = ${model} ===="
mkdir -p ${root}

python ./tools/bonai/bonai_test.py configs/tpami/${model}.py work_dirs/${model}/${epoch}.pth --out ${root}/${model}_${city}_coco_results.pkl --city ${city} --options jsonfile_prefix=${root}/${model} segmentation_eval=False

echo "========== finish ${model} ${city} pkl file generation =========="

python ../bstool/tools/bonai/evaluation_bonai.py --model ${model} --city ${city}

echo "========== finish ${model} ${city} evaluation =========="
