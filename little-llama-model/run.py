from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 下载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 下载数据集
dataset = load_dataset("Fraser/mnist-text-small", split="train", trust_remote_code=True)
sample = dataset[0]["text"]
print("原始样本：", sample)

# 编码输入
inputs = tokenizer(sample, return_tensors="pt")

# 生成结果
outputs = model.generate(**inputs, max_new_tokens=50)
print("生成结果：", tokenizer.decode(outputs[0], skip_special_tokens=True))
