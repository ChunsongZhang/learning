import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


import os

# Ensure the directory exists
os.makedirs('results', exist_ok=True)

class LoRALayer(nn.Module):
    """
    LoRA层实现
    
    LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法，通过添加低秩矩阵来调整预训练模型的权重。
    
    参数:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        r (int): 低秩分解的秩，通常远小于in_features和out_features
        alpha (float): 缩放因子，用于控制LoRA的影响程度
        dropout (float): Dropout概率，用于防止过拟合
        bias (bool): 是否使用偏置项
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # 原始线性层，在推理时使用
        self.original_layer = nn.Linear(in_features, out_features, bias=bias)
        
        # 冻结原始权重
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # LoRA的低秩分解矩阵
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化LoRA权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化LoRA权重"""
        # A初始化为正态分布
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        # B初始化为零，确保训练开始时LoRA不影响原始模型输出
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，例如形状为 [batch_size, in_features]
                如 [32, 768] 表示批次大小为32，每个输入有768个特征
            
        返回:
            torch.Tensor: 输出张量，形状为 [batch_size, out_features]
                如 [32, 512] 表示批次大小为32，每个输出有512个特征
        """
        # 原始层的输出
        original_output = self.original_layer(x)
        
        # LoRA路径的输出
        # 先通过A矩阵，再通过B矩阵，最后应用dropout
        lora_output = self.dropout(self.lora_B(self.lora_A(x)))

        # # 使用F.linear计算LoRA路径的输出
        # output = F.linear(x, self.lora_A.weight @ self.lora_B.weight * self.scaling, bias=self.original_layer.bias)
        # output = self.dropout(output)
        
        # 将原始输出与缩放后的LoRA输出相加
        return original_output + lora_output * self.scaling
    
    def merge_weights(self):
        """
        合并LoRA权重到原始权重中，用于推理加速
        """
        if self.original_layer.weight.requires_grad:
            return
        
        # 计算LoRA的等效权重
        lora_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling

        # 合并到原始权重
        self.original_layer.weight.data += lora_weight

        # 在 PyTorch 中，.data 属性允许你直接访问张量的底层数据，绕过自动求导（autograd）系统
        # 当你需要进行原地（in-place）修改参数而不希望这个操作被记录在计算图中时，使用 .data 是必要的
        # 如果直接使用 self.original_layer.weight += lora_weight：
        #   1. 若权重的 requires_grad=True，会创建不必要的自动求导操作
        #   2. 可能导致后续梯度计算出现意外行为
        #   3. 可能触发 PyTorch 警告，提示正在原地修改需要梯度的张量
        # 使用 .data 可以安全地修改参数值，而不影响梯度计算图
        # 虽然代码中已经检查了 requires_grad，但使用 .data 使意图更明确，并确保了操作的安全性。
        
    def unmerge_weights(self):
        """
        从原始权重中分离LoRA权重，恢复到训练状态
        """
        if self.original_layer.weight.requires_grad:
            return
            
        # 计算LoRA的等效权重
        lora_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        
        # 从原始权重中减去LoRA权重
        self.original_layer.weight.data -= lora_weight


class SimpleModel(nn.Module):
    """
    简单模型用于演示LoRA的使用
    """
    def __init__(self, input_dim, hidden_dim, output_dim, use_lora=False, lora_r=4):
        super().__init__()
        
        if use_lora:
            self.fc1 = LoRALayer(input_dim, hidden_dim, r=lora_r)
            self.fc2 = LoRALayer(hidden_dim, output_dim, r=lora_r)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 测试用例
def test_lora_advantage():
    """
    测试LoRA的优势：参数效率和性能
    """
    print("开始LoRA优势测试...")
    
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成一些合成数据
    input_dim = 100
    hidden_dim = 64
    output_dim = 10
    num_samples = 1000
    
    # 创建随机数据
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 创建标准模型和LoRA模型
    standard_model = SimpleModel(input_dim, hidden_dim, output_dim, use_lora=False)
    lora_model = SimpleModel(input_dim, hidden_dim, output_dim, use_lora=True, lora_r=4)
    
    # 计算参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    standard_params = count_parameters(standard_model)
    lora_params = count_parameters(lora_model)
    
    print(f"标准模型可训练参数数量: {standard_params}")
    print(f"LoRA模型可训练参数数量: {lora_params}")
    print(f"参数减少比例: {(standard_params - lora_params) / standard_params * 100:.2f}%")
    
    # 训练两个模型
    criterion = nn.CrossEntropyLoss()
    standard_optimizer = optim.Adam(standard_model.parameters(), lr=0.001)
    lora_optimizer = optim.Adam(lora_model.parameters(), lr=0.001)
    
    epochs = 10
    standard_losses = []
    lora_losses = []
    
    for epoch in range(epochs):
        # 训练标准模型
        standard_model.train()
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            standard_optimizer.zero_grad()
            outputs = standard_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            standard_optimizer.step()
            epoch_loss += loss.item()
        standard_losses.append(epoch_loss / len(dataloader))
        
        # 训练LoRA模型
        lora_model.train()
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            lora_optimizer.zero_grad()
            outputs = lora_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            lora_optimizer.step()
            epoch_loss += loss.item()
        lora_losses.append(epoch_loss / len(dataloader))
        
        print(f"Epoch {epoch+1}/{epochs} - 标准模型损失: {standard_losses[-1]:.4f}, LoRA模型损失: {lora_losses[-1]:.4f}")
    
    # 评估模型
    standard_model.eval()
    lora_model.eval()
    
    with torch.no_grad():
        standard_outputs = standard_model(X)
        lora_outputs = lora_model(X)
        
        standard_preds = torch.argmax(standard_outputs, dim=1)
        lora_preds = torch.argmax(lora_outputs, dim=1)
        
        standard_accuracy = (standard_preds == y).float().mean().item()
        lora_accuracy = (lora_preds == y).float().mean().item()
    
    print(f"标准模型准确率: {standard_accuracy:.4f}")
    print(f"LoRA模型准确率: {lora_accuracy:.4f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    
    plt.plot(range(1, epochs+1), standard_losses, label='Standard Model')
    plt.plot(range(1, epochs+1), lora_losses, label='LoRA Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison between Standard Model and LoRA Model')
    plt.legend()
    plt.savefig('results/lora_vs_standard_loss.png')
    plt.close()
    
    print("LoRA测试完成！")
    print("结论：")
    print(f"1. LoRA模型可训练参数减少了 {(standard_params - lora_params) / standard_params * 100:.2f}%")
    print(f"2. LoRA模型与标准模型的准确率：{lora_accuracy:.4f} vs {standard_accuracy:.4f}")
    print("3. LoRA适合于资源受限的环境，特别是在微调大型预训练模型时，LoRA的参数效率更高，但是在小模型上，LoRA的性能表现较差")


if __name__ == "__main__":
    test_lora_advantage()

