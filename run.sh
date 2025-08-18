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
export WANDB_USER_KEY=6800d2a81420c3adf2b8f658e79f63bd4003b3e1
export HUGGINGFACE_TOKEN=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
export TORCH_CPP_LOG_LEVEL=ERROR

# Torchrun command
torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=10.42.0.52:29500 \
  diloco_training/training/main.py \
    --model gpt-neo-x \
    --dataset c4 \
    --local_steps 2 \
    --lr 4e-4 \
    --outer_lr 0.7 \
    --warmup_steps 1000 \
    --total_steps 8 \
    --per_device_train_batch_size 64 \
    --batch_size 512 \
    --optim_method sgd \
    --checkpoint_interval 4 \
    --wandb_project_name test_sasho \
    --wandb_group test_sasho_2 \
    --master_address 10.42.0.52 \

