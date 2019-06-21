#!/bin/bash

#### local path
RACE_DIR=../best-compare/data
INIT_CKPT_DIR=xlnet_cased_L-24_H-1024_A-16
PROC_DATA_DIR=proc_data/mcscript
MODEL_DIR=experiment/mcscript

#### google storage path
# GS_ROOT=
# S_INIT_CKPT_DIR=${GS_ROOT}/${INIT_CKPT_DIR}
# GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/race
# GS_MODEL_DIR=${GS_ROOT}/experiment/race

# TPU name in google cloud
# TPU_NAME=

python run_race.py \
  --use_tpu=False \
  --num_hosts=1 \
  --num_core_per_host=3 \
  --model_config_path=${INIT_CKPT_DIR}/xlnet_config.json \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --output_dir=${PROC_DATA_DIR} \
  --init_checkpoint=${INIT_CKPT_DIR}/xlnet_model.ckpt \
  --model_dir=${MODEL_DIR} \
  --data_dir=${RACE_DIR} \
  --max_seq_length=512 \
  --max_qa_length=128 \
  --uncased=False \
  --do_train=True \
  --train_batch_size=4 \
  --do_eval=True \
  --eval_batch_size=4 \
  --train_steps=12000 \
  --save_steps=1000 \
  --iterations=1000 \
  --warmup_steps=1000 \
  --learning_rate=2e-5 \
  --weight_decay=0 \
  --adam_epsilon=1e-6 \
  $@
