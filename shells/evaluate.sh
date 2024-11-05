#!/bin/bash

# Bash script to run the evaluation script

python data_processor/data_collection.py \
    --data_path "/data/repos/KnowledgeUpdate/datasets" \
    --data_name_sft "answers_sft" \
    --data_name_base "answers_base" \
    --max_attempts 5 \
    --retry_delay 2 \