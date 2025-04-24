from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
import torch
import numpy as np
from transformers import DataCollatorForLanguageModeling
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
)

print("已创建一个自定义Transformer模型，可以进行完整训练")

# 加载deepseek-v3的tokenizer用于数据处理
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-v3")

# 加载数据集（示例使用wikitext）
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset["train"]

# 定义计算困惑度(PPL)的评估函数
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # 计算交叉熵损失
    loss = np.mean(np.array([
        -np.sum(np.log_softmax(logit, axis=-1) * np.eye(logit.shape[-1])[label], axis=-1)
        for logit, label in zip(logits, labels)
    ]))
    # 困惑度 = exp(损失)
    perplexity = np.exp(loss)
    return {"perplexity": perplexity}

# 数据整理器
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False
# )

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
    prediction_loss_only=False,  # 确保返回预测结果用于计算指标
    evaluation_strategy="steps",  # 定期评估
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dataset["validation"],
    # data_collator=data_collator,
    compute_metrics=compute_metrics,  # 使用困惑度作为评估指标
)

# 开始训练
trainer.train()

# 评估最终模型的困惑度
eval_results = trainer.evaluate()
print(f"最终模型困惑度 (PPL): {eval_results['eval_perplexity']:.2f}")

# 保存模型
torch.save(model.state_dict(), "./final_transformer_model.pt")