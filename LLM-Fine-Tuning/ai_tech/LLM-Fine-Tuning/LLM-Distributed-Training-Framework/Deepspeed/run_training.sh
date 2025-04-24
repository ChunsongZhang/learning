#!/bin/bash

# DeepSpeed多节点分布式训练启动脚本

# 设置环境变量
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}

# 获取可用GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
export GPUS_PER_NODE=$NUM_GPUS

# 训练参数
BATCH_SIZE=${BATCH_SIZE:-4}
EPOCHS=${EPOCHS:-3}
OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
DATASET=${DATASET:-"wikitext"}
DATASET_CONFIG=${DATASET_CONFIG:-"wikitext-2-raw-v1"}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-512}
DS_CONFIG=${DS_CONFIG:-"ds_config.json"}

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 打印训练配置
echo "=== 训练配置 ==="
echo "节点数: $NNODES"
echo "每节点GPU数: $GPUS_PER_NODE"
echo "主节点地址: $MASTER_ADDR"
echo "主节点端口: $MASTER_PORT"
echo "当前节点排名: $NODE_RANK"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "输出目录: $OUTPUT_DIR"
echo "数据集: $DATASET"
echo "数据集配置: $DATASET_CONFIG"
echo "最大序列长度: $MODEL_MAX_LENGTH"
echo "DeepSpeed配置: $DS_CONFIG"
echo "================="

# 使用deepspeed启动分布式训练
deepspeed --num_nodes=$NNODES \
          --num_gpus=$GPUS_PER_NODE \
          --master_addr=$MASTER_ADDR \
          --master_port=$MASTER_PORT \
          --node_rank=$NODE_RANK \
          train_deepspeed.py \
          --batch_size=$BATCH_SIZE \
          --epochs=$EPOCHS \
          --output_dir=$OUTPUT_DIR \
          --dataset=$DATASET \
          --dataset_config=$DATASET_CONFIG \
          --model_max_length=$MODEL_MAX_LENGTH \
          --ds_config=$DS_CONFIG 