#!/bin/bash

# Bash script to run the data collection script with specified arguments

python data_processor/data_collection.py \
    --data_path "/data/repos/KnowledgeUpdate/datasets/txt_data" \
    --model_name_or_path "/data/repos/huggingface/Qwen2.5-1.5B-Instruct-GPTQ-Int8" \
    --chunk_size_by_token 1024 \
    --qa_amount_per_fact 10 \
    --role_amount_per_fact 3 \
    --json_save_dir "datasets/qa_pairs_parallel_new.json" \
    --num_workers 8
