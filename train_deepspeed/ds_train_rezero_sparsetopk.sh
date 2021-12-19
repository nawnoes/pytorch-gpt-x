#!/bin/bash

base_dir=`pwd`

JOB_NAME=deepspeed_rezero_sparsetopk_gpt

deepspeed ${base_dir}/train_rezero_sparsetopk.py \
          --config ${base_dir}/config_rezero_sparsetopk.json \
          --deepspeed_config ${base_dir}/ds_config_rezero_sparsetopk.json \
#          &> ${JOB_NAME}.log
