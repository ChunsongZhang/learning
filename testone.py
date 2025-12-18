# a = 20
# b = 20
 
# if ( a is b ):
#    print ("1 - a 和 b 有相同的标识")
# else:
#    print ("1 - a 和 b 没有相同的标识")
 
# if ( id(a) == id(b) ):
#    print ("2 - a 和 b 有相同的标识")
# else:
#    print ("2 - a 和 b 没有相同的标识")
# import math
# print(math.sqrt(25))
# print(math.log10(100))
# print(max(1,2,3,4,5))
# import random
# print(random.choice([1,2,3,4,5]))
# import math
# print(math.sin(math.pi/6))
# print("This is the first page.")
# print("\f")  # 插入换页符
# print("This is the second page.") 
# l1 = [1, 2, 3, 4, 5]
# l1.append(6)
# print(l1)
# print(max(1,2,3,4,5))
# def xx(a):
#     def wrapper(*args, **kwargs):  # 接收任意数量的位置和关键字参数
#         print("Before function call")
#         result = a(*args, **kwargs)  # 调用原始函数并传递所有参数
#         print("After function call")
#         return result
#     return wrapper

# @xx
# def greet(name, age):
#     print(f"Hello {name}, you are {age} years old.")

# # 调用装饰后的函数
# greet("Alice", 30)
# def decorator1(func):
#     def wrapper():
#         print("Decorator 1")
#         func()
#     return wrapper

# def decorator2(func):
#     def wrapper():
#         print("Decorator 2")
#         func()
#     return wrapper

# @decorator1
# @decorator2
# def say_hello():
#     print("Hello!")

# say_hello()
# class Test:
#     def prt(self):
#         print(self)
#         print(self.__class__)
 
# t = Test()
# t.prt()
#!/usr/bin/python3
 
# class Vector:
#    def __init__(self, a, b):
#       self.a = a
#       self.b = b
 
#    def __str__(self):
#       return 'Vector (%d, %d)' % (self.a, self.b)
   
#    def __add__(self,other):
#       return Vector(self.a + other.a, self.b + other.b)
 
# v1 = Vector(2,10)
# v2 = Vector(5,-2)
# print (v1 + v2)
#!/usr/bin/python3
 
# num = 1
# def fun1():
#     global num  # 需要使用 global 关键字声明
#     print(num) 
#     num = 123
#     print(num)
# fun1()
# print(num)
# import re

# # 使用 *、+、? 和 {}
# print(re.match(r'\d+?', '12345'))  # 匹配 1 个或多个数字
# print(re.match(r'\d{3}', '12345'))  # 匹配恰好 3 个数字
# print(re.match(r'\d{2,4}?', '12345'))  # 匹配 2 到 4 个数字
# import random
# print(random.choice([1,2,3,4,5]))

# import sys
# print(sys.path)
# mymodule.py
# def greet(name):
#     return f"Hello, {name}!"

# if __name__ == "__main__":
#     print(greet("Alice"))  # 只有当脚本直接运行时才会执行

# import collections
# import hashlib
# import math
# import os
# import random
# import re
# import shutil
# import sys
# import tarfile
# import time
# import zipfile
# from collections import defaultdict
# import pandas as pd
# import requests
# from IPython import display
# from matplotlib import pyplot as plt
# from matplotlib_inline import backend_inline

# d2l = sys.modules[__name__]
# #@save
# import numpy as np
# import torch
# import torchvision
# from PIL import Image
# from torch import nn
# from torch.nn import functional as F
# from torch.utils import data
# from torchvision import transforms
# import torch
# import numpy as np
# print(torch.__version__) # pytorch版本

# # 创建未初始化的张量（垃圾值）
# tensor_a = torch.Tensor()
# print(tensor_a)  # 输出不确定的随机值

# # 创建空张量（元素为0）
# tensor_b = torch.Tensor([2,3])
# print(tensor_b)  # 输出 tensor([])
# print(torch.randn(2,3))

# a = torch.zeros((666,2,3))
# b = torch.zeros((666,3,4))
# print((a @ b).shape)   # 666, 2, 4
 
# a = torch.zeros((2,3))
# b = torch.zeros((666,3,4))
# print((a @ b).shape)  # 666, 2, 4
 
# a = torch.zeros((999,666,2,3))
# b = torch.zeros((999,666,3,4))
# print((a @ b).shape)  # 999, 666, 2, 4
 
# a = torch.zeros((999,1,2,3))
# b = torch.zeros((999,666,3,4))
# print((a @ b).shape)  # 999, 666, 2, 4
 
# a = torch.zeros((666,2,3))
# b = torch.zeros((999,666,3,4))
# print((a @ b).shape)   # 999, 666, 2, 4
 
# a = torch.zeros((2,3))
# b = torch.zeros((999,666,3,4))
# print((a @ b).shape)
# import torch
# import torch.nn as nn

# # 创建一个全连接层，输入特征为 5，输出特征为 3
# linear_layer = nn.Linear(5, 3)

# # 创建一个输入张量，大小为 (batch_size=2, in_features=5)
# x = torch.randn(2, 5)

# 通过线性层进行前向传播
# output = linear_layer(x)

# print("输入张量 x：")
# print(x)
# print("输出张量 output：")
# print(output)
# print("权重矩阵 W：")
# print(linear_layer.weight)  # 权重矩阵，形状 [3, 5]

# print("偏置向量 b：")
# print(linear_layer.bias)    # 偏置，形状 [3]
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import warnings
# warnings.simplefilter("ignore")
# print(torch.__version__)

# src_vocab_size = 11#源词典大小
# target_vocab_size = 11#目标词典大小
# num_layers = 6#层数
# seq_length = 12#序列长度

# # let 0 be sos(start operation signal) token and 1 be eos(end operation signal) token
# src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1],
#                         [0, 2, 8, 7, 3, 4, 5, 6, 7, 2, 10, 1]])#源序列
# target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1],
#                            [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]])#目标序列
# print(src.shape, target.shape)#打印源序列和目标序列的形状
# import torch
# def get_len_mask(b: int, max_len: int, feat_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
#     attn_mask = torch.ones((b, max_len, max_len), device=device)
#     for i in range(b):
#         attn_mask[i, :, :feat_lens[i]] = 0
#     return attn_mask.to(torch.bool)


# m = get_len_mask(2, 4, torch.tensor([2, 4]), "cpu")

# # 为了打印方便，转为int   
# m = m.int()
# print(m)
# import torch
# import torch.nn as nn

# # 假设输入张量是一个 batch_size=2, feature_size=4 的张量
# x = torch.randn(2, 4)

# # 创建层归一化对象
# layer_norm = nn.LayerNorm(4)  # 4 是特征的维度

# # 应用层归一化
# output = layer_norm(x)

# print(output)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class WordEmbedding(nn.Module):
    def __init__(self,vocab_size,embed_size):
        super(WordEmbedding,self).__init__()
        self.embed=nn.Embedding(vocab_size,embed_size)
    def forward(self,x):
        out=self.embed(x)
        return out

class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
        super(PositionalEmbedding,self).__init__()
        self.embed_dim=embed_model_dim
        pe=torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,embed_model_dim,2):
                pe[pos,i]=math.sin(pos/(10000**((2*i)/self.embed_dim)))
                pe[pos,i+1]=math.cos(pos/(10000**((2*(i+1))/self.embed_dim)))
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x*math.sqrt(self.embed_dim)
        seq_len=x.size(1)
        x=x+torch.autograd.Variable(self.pe[:,:seq_len],requires_grad=False)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim=512,n_heads=8):
        super(MultiHeadAttention,self).__init__()
        self.embed_dim=embed_dim
        self.heads=n_heads
        self.single_head_dim=int(self.embed_dim/self.heads)
        self.query_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.key_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.value_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.out=nn.Linear(self.embed_dim,self.embed_dim)
    def forward(self,key,query,value,mask=None):
        batch_size=key.size(0)
        seq_length=key.size(1)
        seq_length_query=query.size(1)
        key=key.view(batch_size,seq_length,self.heads,self.single_head_dim)#将键转换为多头矩阵
        query=query.view(batch_size,seq_length_query,self.heads,self.single_head_dim)#将查询转换为多头矩阵
        value=value.view(batch_size,seq_length,self.heads,self.single_head_dim)#将值转换为多头矩阵
        k=self.key_matrix(key)#将键转换为多头矩阵
        q=self.query_matrix(query)#将查询转换为多头矩阵
        v=self.value_matrix(value)#将值转换为多头矩阵
        
        q=q.transpose(1,2)#将查询转换为多头矩阵
        k=k.transpose(1,2)#将键转换为多头矩阵
        v=v.transpose(1,2)#将值转换为多头矩阵
        #(32 x 8 x 64 x 10)
        k_adjusted=k.transpose(-1,-2)#将键转换为多头矩阵
        #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = (32x8x10x10)
        product=torch.matmul(q,k_adjusted)#计算注意力得分

        if mask is not None:#如果存在掩码
            product=product.masked_fill(mask==0, float("-1e20"))#将掩码位置的注意力得分设置为-1e20

        product=product/math.sqrt(self.single_head_dim)#缩放因子
        scores=F.softmax(product,dim=-1)#计算注意力得分
        #(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64)
        scores=torch.matmul(scores,v)
        concat=scores.transpose(1,2).contiguous().view(batch_size,seq_length_query,self.embed_dim)#将注意力得分转换为多头矩阵

        output=self.out(concat)#将注意力得分转换为多头矩阵

        return output
class TransformerBlock(nn.Module): 
    def __init__(self,embed_dim,expansion_factor=4,n_heads=8):
        super(TransformerBlock,self).__init__()
        self.attention=MultiHeadAttention(embed_dim,n_heads)
        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)

        self.feed_forward=nn.Sequential(
            nn.Linear(embed_dim,expansion_factor*embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor*embed_dim,embed_dim)
        )
        self.dropout1=nn.Dropout(0.2)
        self.dropout2=nn.Dropout(0.2)
    def forward(self,key,query,value):
        attention_out=self.attention(key,query,value)
        attention_residual_out=attention_out+value  
        norm1_out=self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out=self.feed_forward(norm1_out)
        feed_fwd_residual_out=feed_fwd_out+norm1_out
        norm2_out=self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out
    

class TransformerEncoder(nn.Module):
    def __init__(self,seq_len,vocab_size,embed_dim,num_layers=2,expansion_factor=4,n_heads=8):
        super(TransformerEncoder,self).__init__()

        self.embedding_layer=WordEmbedding(vocab_size,embed_dim)
        self.positional_encoder=PositionalEmbedding(seq_len,embed_dim)
        self.layers=nn.ModuleList([TransformerBlock(embed_dim,expansion_factor,n_heads) for i in range(num_layers)])

    def forward(self,x):
        embed_out=self.embedding_layer(x)
        out=self.positional_encoder(embed_out)
        for layer in self.layers:
            out=layer(out,out,out)
        return out
class DecoderBlock(nn.Module):
    def __init__(self,embed_dim,expansion_factor=4,n_heads=8):
        super(DecoderBlock,self).__init__()

        self.attention=MultiHeadAttention(embed_dim,n_heads=8)
        self.norm=nn.LayerNorm(embed_dim)
        #它帮助模型在深层网络中保持稳定的梯度流动，尤其是在使用残差连接的情况下
        self.dropout=nn.Dropout(0.2)
        #随机丢弃20% 神经元防止过拟合
        self.transformer_block=TransformerBlock(embed_dim,expansion_factor,n_heads)
    def forward(self,key,query,x,mask):
        attention=self.attention(x,x,x,mask=mask)
        value=self.dropout(self.norm(attention+x))
        out=self.transformer_block(key,query,value)
        return out
    
class TransformerDecoder(nn.Module):
    def __init__(self,target_vocab_size,embed_dim,seq_len,num_layers=2,expansion_factor=4,n_heads=8):
        super(TransformerDecoder,self).__init__()

        self.word_embedding=WordEmbedding(target_vocab_size,embed_dim)
        self.position_embedding=PositionalEmbedding(seq_len,embed_dim)

        self.layers=nn.ModuleList([DecoderBlock(embed_dim,expansion_factor,n_heads) for _ in range(num_layers)])

        self.fc_out=nn.Linear(embed_dim,target_vocab_size)
        self.dropout=nn.Dropout(0.2)
    
    def forward(self,x,enc_out,mask):
        x=self.word_embedding(x)
        x=self.position_embedding(x)
        x=self.dropout(x)

        for layer in self.layers:
            x=layer(enc_out,x,enc_out,mask)
        out=F.softmax(self.fc_out(x))
        return out
    
class Transformer(nn.Module):
    def __init__(self,embed_dim,src_vocab_size,target_vocab_size,seq_length,num_layers=2,expansion_factor=4,n_heads=8):
        super(Transformer,self).__init__()
        
        self.target_vocab_size=target_vocab_size
        self.encoder=TransformerEncoder(seq_length,src_vocab_size,embed_dim,num_layers=num_layers,expansion_factor=expansion_factor,n_heads=n_heads)
        self.decoder=TransformerDecoder(target_vocab_size,embed_dim,seq_length,num_layers=num_layers,expansion_factor=expansion_factor,n_heads=n_heads)
    
    def mask_trg_mask(self,trg):
        batch_size,trg_len=trg.shape

        trg_mask=torch.tril(torch.ones((trg_len,trg_len))).expand(batch_size,1,trg_len,trg_len) 
        #tril为下三角函数，torch.ones((trg_len,trg_len))创建一个大小为trg_len*trg_len的单位矩阵，   
        # expand(batch_size,1,trg_len,trg_len)将单位矩阵扩展为batch_size*1*trg_len*trg_len的矩阵 
        # 在Transformer模型中，目标序列的掩码（target mask）用于防止模型在解码时看到未来的信息。
        return trg_mask
    
    def decode(self,src,trg):
        trg_mask=self.mask_trg_mask(trg)
        enc_out=self.encoder(src)
        out_labels=[]
        batch_size,seq_len=src.shape[0],src.shape[1]
        
        out=trg

        for i in range(seq_len):
            out=self.decoder(out,enc_out,trg_mask)
            out=out[:,-1,:]
            out=out.argmax(-1)
            out_labels.append(out.item())
            out=torch.unsqueeze(out,axis=0)
        
        return out_labels
    
    def forward(self,src,trg):
        trg_mask=self.mask_trg_mask(trg)
        enc_out=self.encoder(src)
        outputs=self.decoder(trg,enc_out,trg_mask)
        return outputs

if __name__ == '__main__':
    src_vocab_size = 11#源词典大小
    target_vocab_size = 11#目标词典大小
    num_layers = 6#层数
    seq_length = 12#序列长度

    src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1],
                        [0, 2, 8, 7, 3, 4, 5, 6, 7, 2, 10, 1]])#源序列
    target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1],
                           [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]])#目标序列

    print("打印源序列和目标序列的形状", src.shape, target.shape)#打印源序列和目标序列的形状
    model = Transformer(embed_dim=512, src_vocab_size=src_vocab_size,
                        target_vocab_size=target_vocab_size, seq_length=seq_length,
                        num_layers=num_layers, expansion_factor=4, n_heads=8)#创建Transformer模型       
    print("打印模型", model)#打印模型
    out = model(src, target)#前向传播
    print("打印输出", out)#打印输出
    print("打印输出形状", out.shape)#打印输出形状
    print("=" * 50 )#打印分隔符


    # 推理
    src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1]])#源序列
    trg = torch.tensor([[0]])#目标序列
    print("打印源序列和目标序列的形状", src.shape, trg.shape)#打印源序列和目标序列的形状
    print("hello")
    out = model.decode(src, trg)#解码
    print("打印输出", out)#打印输出
    
    

