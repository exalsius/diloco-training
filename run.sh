#!/bin/bash

# Environment variables
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export PYTHONWARNINGS=default
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_USE_V6=0
export GLOO_SOCKET_IFNAME=eth0 
# soeren: this is my wandb api key
export WANDB_USER_KEY=X
export HUGGINGFACE_TOKEN=X
export TORCH_CPP_LOG_LEVEL=ERROR
export HF_HOME="/workspace/datasets"

# Torchrun command
torchrun \
  --nnodes=2 \
  --nproc_per_node=2 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=10.42.0.52:29500 \
  diloco_training/training/main.py \
    --model gpt-neo-x \
    --dataset c4 \
    --local_steps 128 \
    --lr 4e-4 \
    --outer_lr 0.7 \
    --warmup_steps 1000 \
    --total_steps 30000 \
    --per_device_train_batch_size 512 \
    --batch_size 512 \
    --optim_method sgd \
    --checkpoint_interval 512 \
    --wandb_project_name SPRIND \
    --wandb_group SPRIND-c4-gpt-neo-x-diloco-heterogeneous \
    --heterogeneous True \
    --master_address 10.42.0.52 \

