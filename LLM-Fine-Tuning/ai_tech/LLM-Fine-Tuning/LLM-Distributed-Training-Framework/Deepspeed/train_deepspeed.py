import os
import argparse
import torch
import deepspeed
import json
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    get_scheduler
)
from datasets import load_dataset
import sys
sys.path.append("../")
from Transformer.transformer import Transformer

# 命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="使用DeepSpeed ZeRO-2训练DeepSeek风格的Transformer模型")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地GPU排名，DeepSpeed自动设置")
    parser.add_argument("--batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--dataset", type=str, default="wikitext", help="数据集名称")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="数据集配置")
    parser.add_argument("--model_max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--ds_config", type=str, default="ds_config.json", help="DeepSpeed配置文件")
    return parser.parse_args()

# 自定义DeepSeek风格的Transformer模型
class DeepSeekTransformer(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size=2048, num_hidden_layers=24, 
                 num_attention_heads=16, intermediate_size=8192, max_position_embeddings=2048):
        super(DeepSeekTransformer, self).__init__()
        
        # 使用我们的Transformer实现
        self.transformer = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=hidden_size,
            num_heads=num_attention_heads,
            num_layers=num_hidden_layers,
            d_ff=intermediate_size,
            max_seq_length=max_position_embeddings
        )
        
        # 损失计算
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, input_ids, labels=None):
        # 使用自回归方式，input_ids作为源和目标
        # 对于语言模型，我们通常使用input_ids[:-1]作为源，input_ids[1:]作为目标
        if labels is None:
            labels = input_ids.clone()
        
        # 向前传播，注意由于Transformer期望src和tgt，所以我们使用相同的输入
        outputs = self.transformer(input_ids, input_ids)
        
        # 计算损失
        if labels is not None:
            # 重塑输出以适合损失计算
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # 计算损失，需要将输出展平以适应CrossEntropyLoss要求
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            return {"loss": loss, "logits": outputs}
        
        return {"logits": outputs}

def main():
    # 解析参数
    args = parse_args()
    
    # 初始化DeepSpeed
    deepspeed.init_distributed()
    
    # 加载DeepSeek分词器
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
    
    # 创建自定义DeepSeek模型
    model = DeepSeekTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_size=2048,           # 根据需要调整
        num_hidden_layers=24,       # 可根据资源调整
        num_attention_heads=16,     # 注意力头数量
        intermediate_size=8192,     # 前馈网络大小
        max_position_embeddings=args.model_max_length
    )
    
    # 加载数据集
    dataset = load_dataset(args.dataset, args.dataset_config)
    
    # 数据预处理函数
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.model_max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    # 对数据集进行预处理
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 使用因果语言建模而非掩码语言建模
    )
    
    # 读取DeepSpeed配置
    with open(args.ds_config, "r") as f:
        ds_config = json.load(f)
    
    # 确保ZeRO-2级别
    if "zero_optimization" in ds_config:
        if ds_config["zero_optimization"].get("stage", 0) != 2:
            print("警告：DeepSpeed配置未使用ZeRO-2，已自动调整")
            ds_config["zero_optimization"]["stage"] = 2
    else:
        print("警告：DeepSpeed配置中未找到ZeRO优化设置，已自动添加")
        ds_config["zero_optimization"] = {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8
        }
    
    # 初始化DeepSpeed引擎
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # 创建学习率调度器
    num_training_steps = len(tokenized_datasets["train"]) // args.batch_size * args.epochs
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # 训练循环
    device = model_engine.device
    model_engine.train()
    
    for epoch in range(args.epochs):
        train_dataloader = torch.utils.data.DataLoader(
            tokenized_datasets["train"],
            batch_size=args.batch_size,
            collate_fn=data_collator,
            shuffle=True
        )
        
        for step, batch in enumerate(train_dataloader):
            # 将批次移至设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model_engine(input_ids=batch["input_ids"], labels=batch["input_ids"])
            loss = outputs["loss"]
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()
            
            # 打印进度
            if step % 100 == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")
        
        # 每个epoch结束后保存模型
        model_engine.save_checkpoint(args.output_dir, f"epoch_{epoch}")
        print(f"Epoch {epoch} 完成，模型已保存")
    
    print("训练完成！")

if __name__ == "__main__":
    main() 