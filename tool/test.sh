#!/bin/sh

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate pt
PYTHON=python

TEST_CODE=test_strukton.py

dataset=$1
exp_name=$2
leave_out=$3

exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best

now=$(date +"%Y%m%d_%H%M%S")
cp ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}

#: '
$PYTHON -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  DATA.leave_out ${leave_out} \
  save_folder ${result_dir}/best \
  model_path ${model_dir}/model_best.pth \
  2>&1 | tee ${exp_dir}/test_best-$now.log
#'

#: '
$PYTHON -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  DATA.leave_out ${leave_out} \
  save_folder ${result_dir}/last \
  model_path ${model_dir}/model_last.pth \
  2>&1 | tee ${exp_dir}/test_last-$now.log
#'
