# DeepSpeed分布式训练框架

本项目使用DeepSpeed框架实现基于Transformer架构的大型语言模型(LLM)分布式训练，采用ZeRO-2优化策略和DeepSeek模型结构与分词器。

## 目录结构

```
LLM-Training-Framework/
├── Deepspeed/
│   ├── train_deepspeed.py      # 训练脚本
│   ├── ds_config.json          # DeepSpeed配置文件
│   ├── run_training.sh         # 启动训练的Shell脚本
│   └── README.md               # 本文档
```

## DeepSpeed安装教程

### 1. 基础依赖安装

首先安装PyTorch和其他基础依赖：

```bash
conda create -n deepspeed python=3.10
conda activate deepspeed
pip install torch==2.00 torchvision==0.15.1 torchaudio==2.0.1
pip install transformers datasets
```

### 2. 安装DeepSpeed

#### 2.1 简单安装

对于大多数用户，直接通过pip安装即可：

```bash
pip install deepspeed
```

#### 2.2 源码安装（获取最新功能）

如需最新功能或自定义安装，可使用源码安装：

```bash
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
pip install -e .
```

#### 2.3 验证安装

安装完成后，验证DeepSpeed是否安装成功：

```bash
ds_report
```

这将显示DeepSpeed的版本、支持的功能以及CUDA设备信息。

## ZeRO-2分布式训练教程

### 1. ZeRO-2技术简介

ZeRO（Zero Redundancy Optimizer）是DeepSpeed的核心优化技术之一，通过消除训练过程中的内存冗余来支持更大规模的模型训练。

ZeRO-2主要优化：
- 分片优化器状态（如Adam的动量和方差）
- 分片梯度
- 保持模型参数完整

### 2. 配置DeepSpeed ZeRO-2

本项目已提供`ds_config.json`配置文件，主要配置项说明：

```json
{
    "zero_optimization": {
        "stage": 2,                    # 使用ZeRO-2
        "contiguous_gradients": true,  # 使用连续内存梯度
        "overlap_comm": true,          # 通信和计算重叠
        "reduce_scatter": true,        # 使用reduce-scatter操作
        "reduce_bucket_size": 5e8,     # 通信桶大小
        "allgather_bucket_size": 5e8   # 聚合桶大小
    },
    "fp16": {
        "enabled": true                # 启用混合精度训练
    }
}
```

## DeepSpeed训练入门教程（todo）