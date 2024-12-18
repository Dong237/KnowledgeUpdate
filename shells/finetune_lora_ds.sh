#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

#  # Set the path if you do not want to load from huggingface directly
# MODEL="/data/repos/huggingface/Qwen2.5-1.5B-Instruct-GPTQ-Int8" # "/data/repos/huggingface/gpt2"
MODEL="/data/tangbo/plms/Qwen2.5-7B-Instruct-GPTQ-Int8" 
DATA="/data/youxiang/repos/KnowledgeUpdate/datasets/qa_pairs_generated.json"
DS_CONFIG_PATH="ds_config_zero2.json"
WANDB_KEY=""

function usage() {
    echo '
Usage: bash finetune/finetune_lora_ds.sh 
[-m MODEL_PATH] [-d DATA_PATH] [--deepspeed DS_CONFIG_PATH] [--wandb_key WANDB_KEY]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        --deepspeed )
            shift
            DS_CONFIG_PATH=$1
            ;;
        --wandb_key )
            shift
            WANDB_KEY=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --key "e28afd6154b7ecd865dde62fead55bba5994bc9a"\
    --use_wandb True \
    --wandb_run_name "run-lora-20epochs" \
    --data_path $DATA \
    --bf16 True \
    --output_dir output_qwen/7B_20epochs \
    --num_train_epochs 20 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --validation True \
    --validation_size 1000 \
    --logging_strategy "steps" \
    --logging_steps 4 \
    --eval_strategy "steps" \
    --eval_steps 4 \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --report_to "none" \
    --model_max_length 512 \
    --lazy_preprocess True \
    --use_lora \
    --gradient_checkpointing \
    --deepspeed ${DS_CONFIG_PATH} \