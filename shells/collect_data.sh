#!/bin/bash

export CUDA_VISIBLE_DEVICES="1,2,3"

# Bash script to run the data collection script with specified arguments
# --role_amount_per_fact has an impact on processing speed, unrecommended to set it too HIGH
# --chunk_size_by_token defines granularity, has an impact on memory usage, unrecommended to set it too LOW


python data_processor/collect_data_local.py \
    --data_path "/data/youxiang/repos/KnowledgeUpdate/datasets/txt_data" \
    --model_name_or_path "/data/tangbo/plms/Qwen2.5-7B-Instruct/" \
    --chunk_size_by_token 1024 \
    --qa_amount_per_fact 10 \
    --role_amount_per_fact 2 \
    --json_save_dir "datasets/qa_pairs_local.json" \


# # This is a command for collecting data with concurrent processing using an LLM API
# # We do NOT encourage this method, unless you are confident with the API rate limit 
# python data_processor/collect_data_api.py \
#     --data_path "/data/youxiang/repos/KnowledgeUpdate/datasets/txt_data_sample" \
#     --chunk_size_by_token 1024 \
#     --qa_amount_per_fact 10 \
#     --role_amount_per_fact 2 \
#     --json_save_dir "datasets/qa_pairs_parallel.json" \