#!/bin/bash

base_dir=`pwd`

JOB_NAME=deepspeed_gpt3

deepspeed --include localhost:1\
          ${base_dir}/ds_run_pretraining_functional.py \
          --config ${base_dir}/config.json \
          --deepspeed \
          --deepspeed_config ${base_dir}/ds_config.json \
          &> ${JOB_NAME}.log
