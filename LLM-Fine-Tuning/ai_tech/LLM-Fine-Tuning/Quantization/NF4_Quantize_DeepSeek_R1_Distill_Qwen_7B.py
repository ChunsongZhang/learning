# ⚠️ 重要提示 ⚠️
# 注意：
# 量化模型，由于bitsandbytes库的问题，无法在Mac上运行
# 如果您使用的是Mac，请考虑使用其他工具或在云端GPU环境中运行

import torch
import torch.nn as nn
import bitsandbytes as bnb
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 量化配置
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

print(f"\n开始加载{model_name}模型...")
    
try:
    # 尝试以较低精度加载模型以节省内存
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",           # 自动选择设备
        low_cpu_mem_usage=True,      # 低内存使用模式
        quantization_config=nf4_config # 使用nf4量化配置
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"模型 {model_name} NF4量化 加载成功！")
    
    # 分析并打印模型参数信息
    print("\n量化模型参数详情:")
    print("-" * 80)
    print(f"{'参数名':<60} {'形状':<20} {'类型':<10}")
    print("-" * 80)
    
    total_params = 0
    param_types = {}
    
    for name, param in model.named_parameters():
        shape_str = str(tuple(param.shape))
        dtype_str = str(param.dtype).split(".")[-1]
        
        # 计算参数数量
        num_params = np.prod(param.shape)
        total_params += num_params
        
        # 统计不同类型的参数
        if dtype_str not in param_types:
            param_types[dtype_str] = 0
        param_types[dtype_str] += num_params
        
        # 打印参数信息
        print(f"{name:<60} {shape_str:<20} {dtype_str:<10}")
    
    # 打印总结信息
    print("\n" + "=" * 50)
    print(f"模型总参数量: {total_params:,} ({total_params/1e9:.2f}B)")
    
    # 打印不同类型参数的统计
    print("\n参数类型分布:")
    for dtype, count in param_types.items():
        percentage = (count / total_params) * 100
        print(f"  {dtype}: {count:,} ({percentage:.2f}%)")
    
    print("=" * 50)
    
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    print("可能原因: 内存不足或模型路径错误")
    print("建议: 尝试使用更小的模型或在云端GPU环境中运行")