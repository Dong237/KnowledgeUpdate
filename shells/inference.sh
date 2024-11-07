#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=5
DIR=`pwd`

# Run inference.py with specified arguments
python inference.py \
    --model_name_or_path "/data/tangbo/plms/Qwen2.5-7B-Instruct-GPTQ-Int8" \
    --test_data_path "datasets/test_dataset.json" \
    --output_path "results/inference_results_sft_5epochs.json" \
    --lora_weights "/data/youxiang/repos/KnowledgeUpdate/output_qwen/7B_5epochs"

