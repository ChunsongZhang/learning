import torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from Transformer.transformer import Transformer

# 检查是否支持bf16
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    print("BF16混合精度训练可用！")
else:
    print("当前设备不支持BF16混合精度训练，将使用FP32。")

# 创建自定义Transformer模型
src_vocab_size = 128000  # deepseek-v3词表大小
tgt_vocab_size = 128000  # deepseek-v3词表大小
d_model = 768          # 模型维度
num_heads = 12         # 注意力头数
num_layers = 12        # 编码器/解码器层数
d_ff = 3072           # 前馈网络维度
max_seq_length = 512   # 最大序列长度

# 创建Transformer模型
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_length=max_seq_length,
    dropout=0.1
).cuda()

print("已创建一个自定义Transformer模型，可以进行完整训练")

# 加载deepseek-v3的tokenizer用于数据处理
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-v3")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# 在混合精度训练中，使用梯度缩放器 (GradScaler) 来防止数值下溢
scaler = GradScaler()

# 加载数据集（示例使用wikitext）
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset["train"]

# 配置训练参数
batch_size = 4
num_epochs = 10
logging_steps = 100
save_steps = 500

# 准备数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        # 处理输入数据
        text = batch['text']
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
        src = inputs['input_ids'].cuda()
        tgt = src.clone()  # 对于自回归任务，目标就是输入本身，但需要偏移一位
        
        optimizer.zero_grad()

        # 1) 在 autocast 上下文中执行前向与反向计算，并指定 dtype=torch.bfloat16
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # 前向计算使用 bf16
            outputs = model(src, tgt)
            # 计算损失，这里也使用了bf16用于计算Loss（忽略padding的token）
            loss = criterion(outputs.view(-1, tgt_vocab_size), tgt.view(-1))

        # 2) 使用梯度缩放器来反向传播并更新参数
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        
        if (batch_idx + 1) % logging_steps == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx+1}, Loss: {loss.item():.4f}")
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch: {epoch}")
    print(f"训练损失: {avg_train_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "./final_transformer_model.pt")
