### 开源模型量化实战

  **DeepSeek-R1-Distill-Qwen-7B模型量化实战**

  **环境依赖**

  
    conda create --name quantize_env python=3.10
    conda activate quantize_env

    pip install transformers accelerate bitsandbytes
  

  **量化过程**

  详细代码请参考 `NF4_Quantize_DeepSeek_R1_Distill_Qwen_7B.py`

  - 模型：`DeepSeek-R1-Distill-Qwen-7B`

  - 量化方式：`NF4`

  - 量化配置：`bitsandbytes`

  **量化结果**

  - 原始模型参数结果

    ```bash
    模型参数详情:
    --------------------------------------------------------------------------------
    参数名                                                          形状                   类型        
    --------------------------------------------------------------------------------
    model.embed_tokens.weight                                    (152064, 3584)       float32  
    model.layers.0.self_attn.q_proj.weight                       (3584, 3584)         float32  
    model.layers.0.self_attn.q_proj.bias                         (3584,)              float32  
    model.layers.0.self_attn.k_proj.weight                       (512, 3584)          float32  
    model.layers.0.self_attn.k_proj.bias                         (512,)               float32  
    model.layers.0.self_attn.v_proj.weight                       (512, 3584)          float32  
    model.layers.0.self_attn.v_proj.bias                         (512,)               float32  
    model.layers.0.self_attn.o_proj.weight                       (3584, 3584)         float32  
    model.layers.0.mlp.gate_proj.weight                          (18944, 3584)        float32  
    model.layers.0.mlp.up_proj.weight                            (18944, 3584)        float32  
    model.layers.0.mlp.down_proj.weight                          (3584, 18944)        float32  
    model.layers.0.input_layernorm.weight                        (3584,)              float32  
    model.layers.0.post_attention_layernorm.weight               (3584,)              float32  
    model.layers.1.self_attn.q_proj.weight                       (3584, 3584)         float32  
    model.layers.1.self_attn.q_proj.bias                         (3584,)              float32  
    model.layers.1.self_attn.k_proj.weight                       (512, 3584)          float32  
    model.layers.1.self_attn.k_proj.bias                         (512,)               float32  
    model.layers.1.self_attn.v_proj.weight                       (512, 3584)          float32  
    model.layers.1.self_attn.v_proj.bias                         (512,)               float32  
    model.layers.1.self_attn.o_proj.weight                       (3584, 3584)         float32  
    model.layers.1.mlp.gate_proj.weight                          (18944, 3584)        float32  
    model.layers.1.mlp.up_proj.weight                            (18944, 3584)        float32  
    model.layers.1.mlp.down_proj.weight                          (3584, 18944)        float32  
    model.layers.1.input_layernorm.weight                        (3584,)              float32  
    model.layers.1.post_attention_layernorm.weight               (3584,)              float32  
    model.layers.2.self_attn.q_proj.weight                       (3584, 3584)         float32  
    model.layers.2.self_attn.q_proj.bias                         (3584,)              float32  
    ...
    model.layers.27.mlp.up_proj.weight                           (18944, 3584)        float32   
    model.layers.27.mlp.down_proj.weight                         (3584, 18944)        float32  
    model.layers.27.input_layernorm.weight                       (3584,)              float32  
    model.layers.27.post_attention_layernorm.weight              (3584,)              float32  
    model.norm.weight                                            (3584,)              float32  
    lm_head.weight                                               (152064, 3584)       float32   
    ==================================================
    模型总参数量: 7,615,616,512 (7.62B)

    参数类型分布:
    float32: 7,615,616,512 (100.00%)
    ==================================================
    ```

  - 量化模型参数结果

      ```bash
    模型参数详情:
    --------------------------------------------------------------------------------
    参数名                                                          形状                   类型        
    --------------------------------------------------------------------------------
    model.embed_tokens.weight                                    (152064, 3584)       float16  
    model.layers.0.self_attn.q_proj.weight                       (6422528, 1)         uint8    
    model.layers.0.self_attn.q_proj.bias                         (3584,)              float16  
    model.layers.0.self_attn.k_proj.weight                       (917504, 1)          uint8    
    model.layers.0.self_attn.k_proj.bias                         (512,)               float16  
    model.layers.0.self_attn.v_proj.weight                       (917504, 1)          uint8    
    model.layers.0.self_attn.v_proj.bias                         (512,)               float16  
    model.layers.0.self_attn.o_proj.weight                       (6422528, 1)         uint8    
    model.layers.0.mlp.gate_proj.weight                          (33947648, 1)        uint8    
    model.layers.0.mlp.up_proj.weight                            (33947648, 1)        uint8    
    model.layers.0.mlp.down_proj.weight                          (33947648, 1)        uint8    
    model.layers.0.input_layernorm.weight                        (3584,)              float16  
    model.layers.0.post_attention_layernorm.weight               (3584,)              float16  
    model.layers.1.self_attn.q_proj.weight                       (6422528, 1)         uint8    
    model.layers.1.self_attn.q_proj.bias                         (3584,)              float16  
    model.layers.1.self_attn.k_proj.weight                       (917504, 1)          uint8    
    model.layers.1.self_attn.k_proj.bias                         (512,)               float16  
    model.layers.1.self_attn.v_proj.weight                       (917504, 1)          uint8    
    model.layers.1.self_attn.v_proj.bias                         (512,)               float16  
    model.layers.1.self_attn.o_proj.weight                       (6422528, 1)         uint8    
    model.layers.1.mlp.gate_proj.weight                          (33947648, 1)        uint8    
    model.layers.1.mlp.up_proj.weight                            (33947648, 1)        uint8    
    model.layers.1.mlp.down_proj.weight                          (33947648, 1)        uint8    
    model.layers.1.input_layernorm.weight                        (3584,)              float16  
    model.layers.1.post_attention_layernorm.weight               (3584,)              float16  
    model.layers.2.self_attn.q_proj.weight                       (6422528, 1)         uint8    
    model.layers.2.self_attn.q_proj.bias                         (3584,)              float16  
    ...
    model.layers.27.mlp.up_proj.weight                           (33947648, 1)        uint8    
    model.layers.27.mlp.down_proj.weight                         (33947648, 1)        uint8    
    model.layers.27.input_layernorm.weight                       (3584,)              float16  
    model.layers.27.post_attention_layernorm.weight              (3584,)              float16  
    model.norm.weight                                            (3584,)              float16  
    lm_head.weight                                               (152064, 3584)       float16  

    ==================================================
    模型总参数量: 4,352,972,288 (4.35B)

    参数类型分布:
      float16: 1,090,328,064 (25.05%)
      uint8: 3,262,644,224 (74.95%)
    ==================================================
    ```

