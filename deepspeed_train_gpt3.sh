#!/bin/bash

base_dir=`pwd`

JOB_NAME=deepspeed_reformer_gpt3

deepspeed --include localhost:1
          ${base_dir}/deepspeed_train_gpt3.py \
          --config ${base_dir}/config.json \
          --log_dir ${base_dir}/logs \
          --deepspeed \
          --deepspeed_config ${base_dir}/deepspeed_config.json \
          &> ${JOB_NAME}.log