## 混合精度训练（Mixed Precision Training）

将部分计算使用 `float16`（半精度，或使用bf16）进行，其他关键部分仍使用 `float32`（单精度）以保持数值稳定性。使用混合精度后，**你可以训练更大的模型、使用更大的 batch size，而且速度更快。**

| 格式 | 符号位 (S) | 指数位 (E) | 尾数位 (M) | 总位数 |
| ---- | ---------- | ---------- | ---------- | ------ |
| FP32 | 1          | 8          | 23         | 32     |
| BF16 | 1          | 8          | 7          | 16     |
| FP16 | 1          | 5          | 10         | 16     |

### 实现混合精度训练有两种方式

- **在纯 PyTorch 中手动管理**混合精度及训练循环

  可参考代码`pytorch_mixed_precision_training.py`

  ```python
  # 在混合精度训练中，使用梯度缩放器 (GradScaler) 来防止数值下溢
  scaler = GradScaler()
  
  # 在 autocast 上下文中执行前向与反向计算，并指定 dtype=torch.bfloat16
  with autocast(device_type='cuda', dtype=torch.bfloat16):
      output = model(input_data)
      loss = criterion(output, target)
  
  # 使用梯度缩放器来反向传播并更新参数
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  ```

- **使用 Hugging Face Transformers 框架的高级接口**来启用混合精度训练

  可参考代码`huggingface_api_mixed_precision_training.py`

  `bf16=True` 就是开启 **混合精度训练的关键参数**。Transformers 库会在底层使用 PyTorch 的 AMP 机制，并完成 BF16 autocast 与梯度缩放（GradScaler）的管理。

  ```python
  model = AutoModelForCausalLM.from_pretrained("gpt4")
  
  # 开启混合精度训练
  training_args = TrainingArguments(
      output_dir="./output",
      per_device_train_batch_size=8,
      num_train_epochs=3,
      learning_rate=5e-5,
      bf16=True ,  # ✅ 启用混合精度 也可以使用 fp16=True
      logging_steps=10,
  )
  
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=your_dataset,
  )
  
  trainer.train()
  ```

<br>

## 混合精度训练中不同精度(fp32 / fp16)的用途

### 使用float16精度的部分

- **前向传播计算**
  - 模型大部分权重的存储
    
    - 绝大部分模型的可训练参数在混合精度下会保持在 FP16 格式参与前向计算

    - 一些框架（例如 PyTorch AMP 或 TensorFlow的mixed_float16策略）会自动管理和保持一份 FP32 的“主副本”（**master copy**），以防止由于累积数值误差导致的精度下降。在训练时，这份 FP32 master copy 不直接参与前向计算，但在进行优化器更新时，需要用 FP32 master copy 来进行权重的更新操作

  - 大部分激活值
    
    - 一般在前向推理中，激活值也会使用 FP16 进行计算，以提升整体效率。如果某些操作对数值范围比较敏感，或者会导致溢出（overflow/underflow）问题，可能会在框架内部自动将它们切换回 FP32 以保证稳定性

    - 在大多数混合精度训练框架中，即使模型前向传播使用 FP16，在计算损失时会自动将激活值转换为FP32，然后进行损失计算

  - 线性层（全连接层）、卷积层（CNN 模型）、多头注意力层（Transformer 模型）等

  - 不同网络结构会包含不同的算子序列
    
- **反向传播的梯度计算**
  - 为了避免梯度出现溢出/下溢的情况，常常需要使用“梯度缩放”（gradient scaling）技巧：

    - 在 FP16 中，数值范围小，容易出现 underflow（下溢），即梯度接近 0

    - 在反向传播前，先将损失函数值乘以一个放大系数（scale，如 128、1024），让所有梯度都随着损失值一起被放大，远离下溢危险区
      
    - 在反向传播完成后，再对梯度除以相同的放大系数（在优化器 step 前再除回来），恢复到正确的缩放级别
      
    - 梯度缩放（gradient scaling）用于防止梯度下溢

### 保持float32精度的部分

- **优化器状态（如Adam的动量和方差）**
  
  - 优化器通常需要维护一些额外状态，如动量（momentum）、二阶动量（RMSProp / Adam 中的 second moment）等
    
  - 这些状态具有累积性和较长的数值范围，若仅用 FP16 存储，容易在训练后期产生数值误差
    
- **Batch Normalization/Layer Normalization** 层的统计数据（均值和方差）
  
  - BN/LN 等归一化操作在某些情况下对数值精度和稳定性较为敏感，一些实现里会强制这些归一化层使用 FP32 来累积统计量，以尽量减小数值不稳定导致的收敛问题
    
- **损失函数计算**
  
  - 损失函数计算通常涉及数值敏感操作（如log、exp、除法等）
    
  - 损失函数是训练优化的核心指标，精度不足会直接影响模型收敛质量
    
- **主权重副本（master copy of weights）**
  
- **梯度累积**
  
- **梯度更新（optimizer step）**
  
  - 在混合精度训练中（无论是 FP16 还是 BF16），深度学习框架一般会在内部保留一个 FP32 精度的“主权重”用来更新参数，以避免低精度带来的累积误差或溢出
    
- **某些对精度要求高的参数（如嵌入层）**

<br>

## 模型训练时使用Validation Set进行评估

**评估 (Evaluation)** 是指 `Trainer` 内部或手动调用 `trainer.evaluate()` 时，对验证集 (evaluation/validation set) 或测试集 (test set) 做前向推断并**计算指标**的过程。与训练阶段相比，评估并不需要反向传播与参数更新

- 对于大语言模型，评估指标可能是困惑度（Perplexity），即在验证集上跑一次 forward pass，并计算损失，进而得到 Perplexity
  
- 也可在其他任务场景中计算准确率、F1、BLEU 等指标

在配置 `TrainingArguments` 时，设置 `bf16_full_eval=True` 代表 **在评估时也使用 BF16** 来执行前向计算。可参考代码`huggingface_api_mixed_precision_training.py`

**评估中使用 BF16 部分包括**

- 如果前向推理的算子支持 BF16，一般也会用 BF16 来存储和计算
  
- 线性层（matrix multiply / fully-connected）、注意力机制中的矩阵乘法等
  
- 激活张量 (activations)、中间层的激活输出

**评估中使用 FP32 部分包括**

- LayerNorm/BatchNorm 的统计量计算（有些实现会在 FP32 里累计，防止溢出或损失精度）
  
- 不支持低精度的稀有算子（如果模型中有特殊操作）

**TrainingArguments 部分参数解释**

```python
# 配置训练参数，启用bf16混合精度
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    bf16=True,  # 启用bf16混合精度训练
    bf16_full_eval=True,  # 评估时也使用bf16
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
)
```

- **`logging_steps=100`**
  
  - 表示每训练 **100 个优化步骤** (training steps) 进行一次日志记录 (logging)
    
  - 输出内容通常包含当前的 global step、学习率、损失值等信息
    
  - 这些日志信息会写到指定的 `logging_dir`，也可以在命令行或 notebook 的输出里看到

- **`save_steps=500`**
  
  - 表示每训练 **500 个优化步骤** 就自动保存一次模型检查点 (checkpoint)
    
  - 这些检查点包括当前模型权重、优化器状态等，保存在 `output_dir` 对应的子目录下，以便断点续训或回溯
    
- **`eval_steps=500`**
  
  - 表示每训练 **500 个优化步骤** 进行一次评估 (evaluation)
    
  - 评估通常会在 `eval_dataset` 上运行前向计算并计算相应指标或损失值，以便观察模型在验证集上的效果
    
  - 此外，还可以设置 `evaluation_strategy` 参数，`evaluation_strategy` 默认为 `"no"`（即不做自动评估），或者有时是 `"steps"`，由 `eval_steps` 的存在推断出使用 `"steps"`。如果想在**每个 epoch**结束时都做评估，需要把 `evaluation_strategy` 显式设置为 `"epoch"`，这样在每个 epoch 的最后都会评估一次
    
- **`save_total_limit=2`**
  
  - 表示最多保留 **2 个检查点**。当新的检查点被保存时，如果超过了 2 个，最旧的检查点会被自动删除
    
  - 这个设置可以防止磁盘被大量的历史检查点占用
