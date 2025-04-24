import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码模块
    将位置信息编码成向量加入到词向量中,帮助模型学习序列中词的位置关系
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 创建位置索引
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len] -> [max_len, 1] 
        # 创建分母项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加batch维度
        pe = pe.unsqueeze(0)    # [1, max_len, d_model]
        # 注册为buffer(不参与训练)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]  # [batch_size, seq_len, d_model] + [1, seq_len, d_model] 广播机制进行扩展

class MultiHeadAttention(nn.Module):
    """多头注意力模块
    将输入分成多个头,分别计算注意力后合并,增加模型的表达能力
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 每个头的维度 整除/地板除法
        
        # 定义线性变换层
        self.W_q = nn.Linear(d_model, d_model) # 输入维度为d_model，输出维度为d_model
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """计算缩放点积注意力
        Args:
            Q: [batch_size, num_heads, seq_len, d_k]
            K: [batch_size, num_heads, seq_len, d_k]
            V: [batch_size, num_heads, seq_len, d_k]
            mask: [batch_size, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None: 
            scores = scores.masked_fill(mask == 0, -1e9) # 需要mask的部分：填充部分或解码时不能看到未来信息
            
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, V)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换
        """
        view: 改变张量的形状
        transpose: 交换张量的维度
        contiguous: 返回一个内存连续的有相同数据的张量
        """
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k] -> [batch_size, num_heads, seq_len, d_k] 
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        out = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 重组和线性变换
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # [batch_size, num_heads, seq_len, d_k] -> [batch_size, seq_len, num_heads, d_k] -> [batch_size, seq_len, d_model]    
        return self.W_o(out) # 调用一个全连接层，对输入张量 out 进行线性变换，线性层只对张量的最后一个维度进行操作。这一操作将多个注意力头的输出融合在一起，为后续的网络层提供统一的表示。

class FeedForward(nn.Module):
    """前馈神经网络模块
    包含两个线性变换和一个ReLU激活函数
    """
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    """编码器层
    包含多头自注意力和前馈神经网络
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    """解码器层
    包含掩码多头自注意力、多头交叉注意力和前馈神经网络
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 交叉注意力
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    """Transformer模型
    包含编码器和解码器
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 创建多层编码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 创建多层解码器
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size) # 全连接层，将解码器的输出维度从d_model转换为tgt_vocab_size 
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src, tgt):
        """生成源序列和目标序列的掩码"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
        
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 编码器部分
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
        # 解码器部分
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
        output = self.fc(dec_output) # [batch_size, tgt_seq_len, tgt_vocab_size]
        return output

# 使用示例
if __name__ == "__main__":
    # 模型参数
    src_vocab_size = 5000  # 源词表大小
    tgt_vocab_size = 5000  # 目标词表大小
    d_model = 512         # 模型维度
    num_heads = 8         # 注意力头数
    num_layers = 6        # 编码器/解码器层数
    d_ff = 2048          # 前馈网络维度
    max_seq_length = 100  # 最大序列长度
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length
    )
    
    # 创建示例输入
    src = torch.randint(1, src_vocab_size, (64, 30))  # [batch_size, src_seq_len]
    tgt = torch.randint(1, tgt_vocab_size, (64, 35))  # [batch_size, tgt_seq_len]
    
    # 前向传播
    output = model(src, tgt)
    print(f"输出形状: {output.shape}")  # [batch_size, tgt_seq_len, tgt_vocab_size]
