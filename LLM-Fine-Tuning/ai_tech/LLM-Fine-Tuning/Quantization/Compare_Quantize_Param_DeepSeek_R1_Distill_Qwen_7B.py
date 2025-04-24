# ⚠️ 重要提示 ⚠️
# 注意：
# 量化模型，由于bitsandbytes库的问题，无法在Mac上运行
# 如果您使用的是Mac，请考虑使用其他工具或在云端GPU环境中运行

# 若GPU显存足够，可以同时运行以下两个模型，并比较参数数量和类型分布
# 若GPU显存不足，则分别运行以下两个模型，并比较参数数量和类型分布

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import numpy as np

# 设置环境变量，避免在CPU上使用过多内存
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 定义一个函数来加载和分析模型
def load_and_analyze_model(model_name, model_display_name):
    print(f"\n开始加载{model_display_name}模型...")
    
    try:
        # 尝试以较低精度加载模型以节省内存
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",           # 自动选择设备
            low_cpu_mem_usage=True,      # 低内存使用模式
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"模型 {model_display_name} 加载成功！")
        
        # 分析并打印模型参数信息
        print("\n模型参数详情:")
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
        
        return total_params, param_types
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        print("可能原因: 内存不足或模型路径错误")
        print("建议: 尝试使用更小的模型或在云端GPU环境中运行")
        return 0, {}

# 加载原始模型
original_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
original_params, original_types = load_and_analyze_model(original_model_name, "原始DeepSeek-R1-Distill-Qwen-7B")

# 加载4bit量化模型
quantized_model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit"
quantized_params, quantized_types = load_and_analyze_model(quantized_model_name, "4bit量化的DeepSeek-R1-Distill-Qwen-7B")

# 对比两个模型
if original_params > 0 and quantized_params > 0:
    print("\n\n" + "=" * 80)
    print("模型对比结果:")
    print("-" * 80)
    print(f"原始模型参数量: {original_params:,} ({original_params/1e9:.2f}B)")
    print(f"量化模型参数量: {quantized_params:,} ({quantized_params/1e9:.2f}B)")
    
    # 计算参数量减少百分比
    reduction = (original_params - quantized_params) / original_params * 100
    print(f"参数量减少: {reduction:.2f}%")
    
    # 对比参数类型分布
    print("\n参数类型分布对比:")
    print(f"{'类型':<15} {'原始模型':<25} {'量化模型':<25}")
    print("-" * 80)
    
    all_types = set(list(original_types.keys()) + list(quantized_types.keys()))
    for dtype in all_types:
        orig_count = original_types.get(dtype, 0)
        quant_count = quantized_types.get(dtype, 0)
        
        orig_percentage = (orig_count / original_params) * 100 if original_params > 0 else 0
        quant_percentage = (quant_count / quantized_params) * 100 if quantized_params > 0 else 0
        
        print(f"{dtype:<15} {orig_count:,} ({orig_percentage:.2f}%) {quant_count:,} ({quant_percentage:.2f}%)")
    
    print("=" * 80)
