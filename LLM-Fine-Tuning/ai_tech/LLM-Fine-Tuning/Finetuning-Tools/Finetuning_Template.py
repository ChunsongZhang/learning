import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

class DeepSeekFineTuner:
    """使用LLaMA-Factory微调DeepSeek-70B模型的工具类"""
    
    def __init__(
        self,
        model_name="deepseek-ai/deepseek-llm-70b-base",
        dataset_name="tatsu-lab/alpaca",
        output_dir="./deepseek_finetuned",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        use_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        max_seq_length=512,
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        warmup_ratio=0.03,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=50,
        seed=42
    ):
        """
        初始化DeepSeek微调器
        
        参数:
            model_name: 预训练模型名称
            dataset_name: 数据集名称
            output_dir: 输出目录
            lora_r: LoRA秩
            lora_alpha: LoRA缩放因子
            lora_dropout: LoRA dropout率
            use_4bit: 是否使用4bit量化
            bnb_4bit_compute_dtype: 4bit计算的数据类型
            bnb_4bit_quant_type: 4bit量化类型
            max_seq_length: 最大序列长度
            batch_size: 批次大小
            gradient_accumulation_steps: 梯度累积步数
            learning_rate: 学习率
            num_train_epochs: 训练轮数
            warmup_ratio: 预热比例
            evaluation_strategy: 评估策略
            eval_steps: 评估步数
            save_strategy: 保存策略
            save_steps: 保存步数
            logging_steps: 日志步数
            seed: 随机种子
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_ratio = warmup_ratio
        self.evaluation_strategy = evaluation_strategy
        self.eval_steps = eval_steps
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.seed = seed
        
        # 设置随机种子
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化分词器和模型
        self._init_tokenizer_and_model()
        
        # 加载数据集
        self._load_dataset()
    
    def _init_tokenizer_and_model(self):
        """初始化分词器和模型"""
        print("正在加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("正在加载模型...")
        # 量化配置
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 为量化训练准备模型
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # 配置LoRA
        self.lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
    
    def _load_dataset(self):
        """加载并处理数据集"""
        print("正在加载数据集...")
        self.dataset = load_dataset(self.dataset_name)
        
        # 数据集预处理函数
        def preprocess_function(examples):
            # 构建提示模板
            prompts = []
            for i in range(len(examples["instruction"])):
                if examples["input"][i]:
                    prompt = f"### 指令:\n{examples['instruction'][i]}\n\n### 输入:\n{examples['input'][i]}\n\n### 回答:\n{examples['output'][i]}"
                else:
                    prompt = f"### 指令:\n{examples['instruction'][i]}\n\n### 回答:\n{examples['output'][i]}"
                prompts.append(prompt)
            
            # 分词
            tokenized_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            # 设置标签
            tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
            
            return tokenized_inputs
        
        # 应用预处理
        self.processed_dataset = self.dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        
        # 分割数据集
        if "validation" not in self.processed_dataset:
            self.processed_dataset = self.processed_dataset["train"].train_test_split(
                test_size=0.1, seed=self.seed
            )
            self.train_dataset = self.processed_dataset["train"]
            self.eval_dataset = self.processed_dataset["test"]
        else:
            self.train_dataset = self.processed_dataset["train"]
            self.eval_dataset = self.processed_dataset["validation"]
    
    def train(self):
        """训练模型"""
        print("开始训练模型...")
        
        # 数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            warmup_ratio=self.warmup_ratio,
            evaluation_strategy=self.evaluation_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="tensorboard",
            remove_unused_columns=False,
            fp16=not self.use_4bit,  # 如果使用4bit量化，则不使用fp16
            seed=self.seed
        )
        
        # 初始化训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        self.model.save_pretrained(os.path.join(self.output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_model"))
        
        print(f"训练完成！模型已保存到 {self.output_dir}/final_model")
    
    def evaluate_model(self, test_prompts, original_model_name=None):
        """
        评估微调前后的模型性能
        
        参数:
            test_prompts: 测试提示列表，每个提示是一个字典，包含instruction和input字段
            original_model_name: 原始模型名称，如果为None则使用self.model_name
        """
        print("开始评估模型性能...")
        
        # 加载原始模型
        if original_model_name is None:
            original_model_name = self.model_name
        
        print(f"加载原始模型: {original_model_name}")
        original_tokenizer = AutoTokenizer.from_pretrained(
            original_model_name,
            trust_remote_code=True,
            use_fast=False
        )
        original_model = AutoModelForCausalLM.from_pretrained(
            original_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 加载微调后的模型
        print(f"加载微调后的模型: {self.output_dir}/final_model")
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(self.output_dir, "final_model"),
            device_map="auto",
            trust_remote_code=True
        )
        
        # 评估结果
        results = {
            "original": [],
            "finetuned": []
        }
        
        # 生成回答
        for prompt_data in tqdm(test_prompts, desc="生成回答"):
            # 构建提示
            if prompt_data.get("input", ""):
                prompt = f"### 指令:\n{prompt_data['instruction']}\n\n### 输入:\n{prompt_data['input']}\n\n### 回答:\n"
            else:
                prompt = f"### 指令:\n{prompt_data['instruction']}\n\n### 回答:\n"
            
            # 使用原始模型生成
            inputs = original_tokenizer(prompt, return_tensors="pt").to(original_model.device)
            original_output = original_model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            original_response = original_tokenizer.decode(original_output[0], skip_special_tokens=True)
            original_response = original_response.replace(prompt, "").strip()
            
            # 使用微调后的模型生成
            inputs = self.tokenizer(prompt, return_tensors="pt").to(finetuned_model.device)
            finetuned_output = finetuned_model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            finetuned_response = self.tokenizer.decode(finetuned_output[0], skip_special_tokens=True)
            finetuned_response = finetuned_response.replace(prompt, "").strip()
            
            # 保存结果
            results["original"].append({
                "prompt": prompt,
                "response": original_response
            })
            results["finetuned"].append({
                "prompt": prompt,
                "response": finetuned_response
            })
        
        # 保存评估结果
        with open(os.path.join(self.output_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"评估完成！结果已保存到 {self.output_dir}/evaluation_results.json")
        
        # 返回结果以便进一步分析
        return results
    
    def analyze_results(self, results):
        """
        分析评估结果
        
        参数:
            results: 评估结果，包含original和finetuned两个列表
        """
        print("分析评估结果...")
        
        # 计算平均响应长度
        original_lengths = [len(r["response"].split()) for r in results["original"]]
        finetuned_lengths = [len(r["response"].split()) for r in results["finetuned"]]
        
        avg_original_length = sum(original_lengths) / len(original_lengths)
        avg_finetuned_length = sum(finetuned_lengths) / len(finetuned_lengths)
        
        print(f"原始模型平均响应长度: {avg_original_length:.2f} 词")
        print(f"微调模型平均响应长度: {avg_finetuned_length:.2f} 词")
        
        # 绘制响应长度分布
        plt.figure(figsize=(12, 6))
        plt.hist(original_lengths, alpha=0.5, label="原始模型")
        plt.hist(finetuned_lengths, alpha=0.5, label="微调模型")
        plt.xlabel("响应长度（词数）")
        plt.ylabel("频率")
        plt.title("模型响应长度分布")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "response_length_distribution.png"))
        plt.close()
        
        # 输出一些示例比较
        print("\n示例比较:")
        for i in range(min(3, len(results["original"]))):
            print(f"\n示例 {i+1}:")
            print(f"提示: {results['original'][i]['prompt']}")
            print(f"原始模型: {results['original'][i]['response'][:200]}...")
            print(f"微调模型: {results['finetuned'][i]['response'][:200]}...")


# 使用示例
if __name__ == "__main__":
    # 初始化微调器
    finetuner = DeepSeekFineTuner(
        model_name="deepseek-ai/deepseek-llm-70b-base",
        dataset_name="tatsu-lab/alpaca",
        output_dir="./deepseek_finetuned",
        lora_r=16,
        lora_alpha=32,
        batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3
    )
    
    # 训练模型
    finetuner.train()
    
    # 评估模型
    test_prompts = [
        {
            "instruction": "解释量子计算的基本原理",
            "input": ""
        },
        {
            "instruction": "写一首关于人工智能的诗",
            "input": ""
        },
        {
            "instruction": "比较传统机器学习和深度学习的区别",
            "input": ""
        },
        {
            "instruction": "总结以下文本的要点",
            "input": "深度学习是机器学习的一个分支，它使用多层神经网络来提取数据的高级特征。与传统机器学习方法相比，深度学习可以自动学习特征，无需手动特征工程。近年来，深度学习在计算机视觉、自然语言处理和语音识别等领域取得了突破性进展。"
        }
    ]
    
    results = finetuner.evaluate_model(test_prompts)
    
    # 分析结果
    finetuner.analyze_results(results)

