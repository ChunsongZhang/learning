# ⚠️ 重要提示 ⚠️
# 注意：Unsloth目前仅支持带有Nvidia GPU的PC
# 在Mac上由于Xformers包的依赖问题，无法安装
# 如果您使用的是Mac，请考虑使用其他微调工具或在云端GPU环境中运行

import torch
from unsloth import FastLanguageModel
from peft import PeftModel
import random
from bert_score import score
from tqdm import tqdm
from datasets import load_dataset


# 路径配置------------------------------------------------------------------------
base_model_path = "unsloth/Qwen2.5-7B-Instruct"  # 原始预训练模型路径

#[NOTE] 这里的 lora_path 是在微调训练过程中保存的本地 LoRA 模型路径
peft_model_path = "lora_model"  # LoRA 微调后保存的适配器路径

# 模型和分词器加载------------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B-Instruct", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# 加载 LoRA 适配器（在基础模型上加载微调参数）
lora_model = PeftModel.from_pretrained(
    model,
    peft_model_path,
)

# 合并 LoRA 权重到基础模型（提升推理速度，但会失去再次训练的能力）
lora_model = lora_model.merge_and_unload()
lora_model.eval()  # 设置为评估模式

# 生成函数------------------------------------------------------------------------
def generate_response(model, prompt):
    """
    统一的生成函数

    参数：
    model: 要使用的模型实例
    prompt: 符合格式要求的输入文本

    返回：
    清洗后的回答文本
    """
    # 输入编码（保持与训练时相同的处理方式）
    inputs = tokenizer(
        prompt,
        return_tensors="pt",  # 返回 PyTorch 张量
        max_length=128,  # 最大输入长度（与训练时一致）
        truncation=True,  # 启用截断
        padding="max_length"  # 填充到最大长度（保证 batch 一致性）
    ).to(model.device)  # 确保输入与模型在同一设备

    # 文本生成（关闭梯度计算以节省内存）
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=128,  # 生成内容的最大 token 数（控制回答长度）
            temperature=0.7,  # 温度参数（0.0-1.0，值越大随机性越强）
            top_p=0.9,  # 核采样参数（保留累积概率前 90% 的 token）
            repetition_penalty=1.1,  # 重复惩罚系数（>1.0 时抑制重复内容）
            eos_token_id=tokenizer.eos_token_id,  # 结束符 ID
            pad_token_id=tokenizer.pad_token_id  # 填充符 ID
        )

    # 解码与清洗输出
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  # 跳过特殊 token
    answer = full_text.split("###答案：\n")[-1].strip()  # 提取答案部分
    return answer


# 主程序------------------------------------------------------------------------
if __name__ == "__main__":
    # 加载测试数据
    ####-----------批量测试---------------#

    # 尝试从Hugging Face加载数据集
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "zh")

    # 随机选择10个样本
    random_indices = random.sample(range(len(dataset["train"])), 10)
    random_samples = [dataset["train"][i] for i in random_indices]

    print("成功加载数据集！")
    print(f"数据集总样本数：{len(dataset['train'])}")
    print("以下是随机选择的10个问题：\n")

    # 显示随机选择的10个问题
    for i, sample in enumerate(random_samples):
        question = sample["Question"]
        complex_cot = sample["Complex_COT"]
        response = sample["Response"]
        
        print(f"样本 #{random_indices[i] + 1}")
        print(f"问题 {i+1}:\n{question}\n")
        print(f"复杂推理过程:\n{complex_cot}\n")
        print(f"参考回答:\n{response}")
        print("-" * 50)

    # 数据量比较大，我们只选择 10 条数据进行测试
    test_data = random_samples

    # 批量生成回答
    def batch_generate(model, questions):
        answers = []
        for q in tqdm(questions):
            prompt = f"诊断问题：{q}\n详细分析：\n###答案：\n"
            ans = generate_response(model, prompt)
            answers.append(ans)
        return answers

    # 生成结果
    base_answers = batch_generate(model, [d["Question"] for d in test_data])
    lora_answers = batch_generate(lora_model, [d["Question"] for d in test_data])
    ref_answers = [d["Response"] for d in test_data]

    bert_model_path = "bert-base-chinese"

    # 计算 BERTScore
    _, _, base_bert = score(
        base_answers,
        ref_answers,
        lang="zh",
        model_type=bert_model_path,
        num_layers=12,
    )

    _, _, lora_bert = score(
        lora_answers,
        ref_answers,
        lang="zh",
        model_type=bert_model_path,
        num_layers=12,
    )

    print(f"BERTScore | 原始模型: {base_bert.mean().item():.3f} | LoRA 模型: {lora_bert.mean().item():.3f}")