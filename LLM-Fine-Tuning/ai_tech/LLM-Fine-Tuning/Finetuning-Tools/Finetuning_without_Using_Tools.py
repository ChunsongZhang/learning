import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

# 指定使用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 配置路径（根据实际路径修改）
model_path = "xxxx"  # 模型路径
data_path = "xxxx"   # 数据集路径
output_path = "xxxx" # 微调后模型保存路径

# 设置设备参数
DEVICE = "cuda"
DEVICE_ID = "0"
device = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合 CUDA 设备信息

# 自定义回调记录 Loss
class LossCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])

# 数据预处理函数
def process_data(tokenizer):
    dataset = load_dataset("json", data_files=data_path, split="train[:1500]")

    def format_example(example):
        instruction = f"诊断问题：{example['Question']}\n详细分析：{example['Complex_CoT']}"
        inputs = tokenizer(
            f"{instruction}\n###答案：\n{example['Response']}<|endoftext|>",
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

    return dataset.map(format_example, remove_columns=dataset.column_names)

# LoRA 配置
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 训练参数配置
training_args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=2,  # 显存优化设置
    gradient_accumulation_steps=4,  # 累计梯度相当于 batch_size=8
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=True,  # 开启混合精度
    logging_steps=20,
    save_strategy="no",
    report_to="none",
    optim="adamw_torch",
    no_cuda=False,  # 强制使用 CUDA
    dataloader_pin_memory=False,  # 加速数据加载
    remove_unused_columns=False,  # 防止删除未使用的列
    device="cuda:0"  # 指定使用的 GPU 设备
)

def main():
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载模型到 GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map=device
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 准备数据
    dataset = process_data(tokenizer)

    # 训练回调
    loss_callback = LossCallback()

    # 数据加载器
    def data_collator(data):
        batch = {
            "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]).to(device),
            "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]).to(device),
            "labels": torch.stack([torch.tensor(d["input_ids"]) for d in data]).to(device)  # 使用 input_ids 作为 labels
        }
        return batch

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[loss_callback]
    )

    # 开始训练
    print("开始训练...")
    trainer.train()

    # 保存最终模型
    trainer.model.save_pretrained(output_path)
    print(f"模型已保存至：{output_path}")

    # 绘制训练集损失 Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_callback.losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_path, "loss_curve.png"))
    print("Loss 曲线已保存")

if __name__ == "__main__":
    main()
