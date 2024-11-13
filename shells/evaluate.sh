#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
# Bash script to run the evaluation script

python evaluate.py \
    --data_path "/data/youxiang/repos/KnowledgeUpdate/results" \
    --data_name_sft "inference_results_sft_10epochs.json" \
    --data_name_base "inference_results_base.json" \
    --max_attempts 5 \
    --retry_delay 2 \
    --local_model_path "/data/tangkai/models/Qwen2.5-72B-Instruct-GPTQ-Int4"