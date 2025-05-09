# 🗣️ Kokoro-82M TTS 模型 学习笔记

本笔记整理了使用 Kokoro-82M 语音合成模型（Text-to-Speech，TTS）过程中相关的常见问题与实践经验，供今后复现或参考。

---

## 📦 使用的模型与数据集

- **模型**：Kokoro-82M（轻量级多语种 TTS 模型）
- **英文数据集**：LJSpeech，裁剪约 2000 条样本
- **中文数据集**：AISHELL-3，裁剪约 2000 条样本

---

## 🧹 数据清洗

### 为什么需要清洗？

- 格式统一（如标注文本、音频路径）
- 去除无效或损坏的数据块
- 预处理如繁简转换、去除特殊符号等

### 清洗步骤（示例代码）

> 🔧 文件位置：`tts_data_cleaner/clean.py`

```python
import os
import re
import json
from tqdm import tqdm

def clean_aishell3(input_dir, output_path, max_items=2000):
    result = []
    count = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                transcript_path = file.replace('.wav', '.txt')
                txt_full_path = os.path.join(root, transcript_path)
                wav_full_path = os.path.join(root, file)
                if os.path.exists(txt_full_path):
                    with open(txt_full_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？、]", "", text)
                    result.append((wav_full_path, text))
                    count += 1
                    if count >= max_items:
                        break
    with open(output_path, 'w', encoding='utf-8') as f:
        for wav, txt in result:
            f.write(f"{wav}|{txt}\n")

```
## 🚀 推理和生成流程问题

### 错误提示及解决：

1. ❌ `float() argument must be a string or a real number, not 'generator'`  
    ✅ **解决**：改用 `list()` 展开 generator，再进行拼接。
    
2. ❌ `setting an array element with a sequence...`  
    ✅ **解决**：检查 `KPipeline.Result` 中是否包含不规则结构；统一数据格式或跳过无效片段。
    
3. ❌ `[ERROR] 所有音频块都是空的或不可用`  
    ✅ **解决**：清洗数据集后，重新检查音频路径是否有效，并在 TTS 推理中打印每一块的内容。
## ⏳ tqdm 进度条使用

在清洗或推理过程中推荐使用 `tqdm` 追踪进度：
pip install tqdm
from tqdm import tqdm

for item in tqdm(data):
    process(item)
## 📂 目录结构建议
kokoro_project/
├── tts_data_cleaner/        # 数据清洗脚本
│   └── clean.py
├── datasets/
│   ├── aishell3/            # 原始数据
│   └── aishell3_cleaned.txt # 清洗后的数据列表
├── inference/               # 推理脚本或 notebook
├── model/                   # 下载或保存的模型权重
├── README.md                # 学习笔记和项目说明
## 📌 其他备注

- **数据未清洗是否能运行？**：理论上可以，但模型容易报错或合成效果差。
    
- **建议：** 开发阶段裁剪小数据集（约 2000 条）验证流程，避免加载整个大规模数据集。
## 📚 参考资料

- [Kokoro 模型 GitHub 项目](https://github.com/Plachtaa/kokoro)
    
- LJSpeech 数据集
    
- AISHELL-3 中文语音数据集


