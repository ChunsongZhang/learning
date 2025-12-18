import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WordEmbedding(nn.Module):#词嵌入层
    def __init__(self,vocab_size,embed_size):#初始化，vocab_size是词典大小，embed_size是词向量维度
        super(WordEmbedding,self).__init__()#调用父类nn.Module的初始化方法
        self.embed=nn.Embedding(vocab_size,embed_size)#创建一个词嵌入层，将词索引转换为密集向量
    
    def forward(self,x):#前向传播，x是输入的词索引
        #x:(batch_size,seq_len)
        out=self.embed(x)#将词索引转换为密集向量
        return out#返回词嵌入层的输出

class PositionalEmbedding(nn.Module):#位置编码层
    def __init__(self,max_seq_len,embed_model_dim):#初始化，max_seq_len是最大序列长度，embed_model_dim是词向量维度
        super(PositionalEmbedding,self).__init__()#调用父类nn.Module的初始化方法
        self.embed_dim=embed_model_dim#词向量维度

        pe=torch.zeros(max_seq_len,self.embed_dim)#创建一个位置编码矩阵，大小为max_seq_len*embed_model_dim

        for pos in range(max_seq_len):#遍历每个位置
            for i in range(0,embed_model_dim,2):#遍历每个位置的每个维度
                pe[pos,i]=math.sin(pos/(10000**((2*i)/self.embed_dim)))#计算位置编码
                pe[pos,i+1]=math.cos(pos/(10000**((2*(i+1))/self.embed_dim)))#计算位置编码
        pe=pe.unsqueeze(0) 
        #在深度学习中，通常需要处理批量数据。通过在第 0 维添加一个维度，pe 可以与批量数据的形状相匹配。
        # 这样做的好处是，即使在没有批量处理的情况下（即批量大小为 1），代码仍然可以正常工作。
        self.register_buffer('pe',pe)
        #这是 PyTorch 中 nn.Module 类的方法，用于将一个张量注册为模型的缓冲区。
        #缓冲区是模型的一部分，但在训练过程中不会更新其梯度。

    def forward(self,x):#前向传播，x是输入的词索引
        #x:(batch_size,seq_len,embed_dim)
        x=x*math.sqrt(self.embed_dim)#缩放因子
        seq_len=x.size(1)#x.size(1) 返回张量的第 1 维的大小，即序列长度。
        x=x+torch.autograd.Variable(self.pe[:,:seq_len],requires_grad=False)#将位置编码矩阵添加到输入中
        return x


class MultiHeadAttention(nn.Module):#多头注意力机制
    def __init__(self,embed_dim=512,n_heads=8):#初始化，embed_dim是词向量维度，n_heads是多头注意力机制的头部数量
        super(MultiHeadAttention,self).__init__()#调用父类nn.Module的初始化方法，良好的编程习惯
        #embed_dim=d_model
        self.embed_dim=embed_dim#词向量维度
        self.heads=n_heads#多头注意力机制的头部数量
        self.single_head_dim=int(self.embed_dim/self.heads)#每个头的大小

        self.query_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)#创建一个线性层，将每个头的查询矩阵转换为单头矩阵
        self.value_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)#创建一个线性层，将每个头的值矩阵转换为单头矩阵
        self.key_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)#创建一个线性层，将每个头的键矩阵转换为单头矩阵

        self.out=nn.Linear(self.embed_dim,self.embed_dim)#创建一个线性层，将多头矩阵转换为单头矩阵

    def forward(self,key,query,value,mask=None):#前向传播，key,query,value是输入的键、查询和值，mask是掩码
        batch_size=key.size(0)#批量大小
        seq_length=key.size(1)#序列长度
        # query dimension can change in decoder during inference.
        # so we cant take general seq_length    
        seq_length_query=query.size(1)#查询序列长度

        key=key.view(batch_size,seq_length,self.heads,self.single_head_dim)#将键转换为多头矩阵
        query=query.view(batch_size,seq_length_query,self.heads,self.single_head_dim)#将查询转换为多头矩阵
        value=value.view(batch_size,seq_length,self.heads,self.single_head_dim)#将值转换为多头矩阵

        #print("key,query,value的形状",key.shape,query.shape,value.shape)
        k=self.key_matrix(key)#将键转换为多头矩阵
        q=self.query_matrix(query)#将查询转换为多头矩阵
        v=self.value_matrix(value)#将值转换为多头矩阵
        # print("q的值",q.shape,k.shape,v.shape)

        q=q.transpose(1,2)#将查询转换为多头矩阵
        k=k.transpose(1,2)#将键转换为多头矩阵
        v=v.transpose(1,2)#将值转换为多头矩阵

        k_adjusted=k.transpose(-1,-2)#将键转换为多头矩阵
        #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = (32x8x10x10)
        product=torch.matmul(q,k_adjusted)#计算注意力得分

        if mask is not None:#如果存在掩码
            product=product.masked_fill(mask==0, float("-1e20"))#将掩码位置的注意力得分设置为-1e20

        product=product/math.sqrt(self.single_head_dim)#缩放因子
        #print("product的值",product.shape)
        scores=F.softmax(product,dim=-1)#计算注意力得分
        #print("scores的值",scores.shape)
        scores=torch.matmul(scores,v)#计算注意力得分
        #print("scores的值",scores.shape)
        concat=scores.transpose(1,2).contiguous().view(batch_size,seq_length_query,self.embed_dim)
        # print("concat的值",concat.shape)
        #contiguous()保证了张量的数据在内存中是按顺序存储的，即内存地址是连续的
        #将注意力得分转换为多头矩阵
        output=self.out(concat)#将注意力得分转换为多头矩阵

        return output

class TransformerBlock(nn.Module):#Transformer块，一个编码器块
    def __init__(self,embed_dim,expansion_factor=4,n_heads=8):#初始化，embed_dim是词向量维度，expansion_factor是扩展因子，n_heads是多头注意力机制的头部数量
        super(TransformerBlock,self).__init__()#调用父类nn.Module的初始化方法

        self.attention=MultiHeadAttention(embed_dim,n_heads)#多头注意力机制
        self.norm1=nn.LayerNorm(embed_dim)#层归一化
        self.norm2=nn.LayerNorm(embed_dim)#层归一化

        self.feed_forword=nn.Sequential(
            nn.Linear(embed_dim,expansion_factor*embed_dim),#创建一个线性层，将词向量维度扩展为expansion_factor*embed_dim
            nn.ReLU(),#ReLU激活函数
            nn.Linear(expansion_factor*embed_dim,embed_dim)#创建一个线性层，将词向量维度扩展为embed_dim
        )

        self.dropout1=nn.Dropout(0.2)#Dropout层
        self.dropout2=nn.Dropout(0.2)#Dropout层
    
    def forward(self,key,query,value):#前向传播，key,query,value是输入的键、查询和值
        attention_out=self.attention(key,query,value)#多头注意力机制
        attention_residual_out=attention_out+value#残差连接
        norm1_out=self.dropout1(self.norm1(attention_residual_out))#层归一化

        feed_fwd_out=self.feed_forword(norm1_out)#前馈神经网络  
        feed_fwd_residual_out=feed_fwd_out+norm1_out#残差连接
        norm2_out=self.dropout2(self.norm2(feed_fwd_residual_out))#层归一化

        return norm2_out

class TransformerEncoder(nn.Module):#Transformer编码器
    def __init__(self,seq_len,vocab_size,embed_dim,num_layers=2,expansion_factor=4,n_heads=8):#初始化，seq_len是序列长度，vocab_size是词典大小，embed_dim是词向量维度，num_layers是层数，expansion_factor是扩展因子，n_heads是多头注意力机制的头部数量
        super(TransformerEncoder,self).__init__()#调用父类nn.Module的初始化方法

        self.embedding_layer=WordEmbedding(vocab_size,embed_dim)#词嵌入层
        self.positional_encoder=PositionalEmbedding(seq_len,embed_dim)#位置编码层
        self.layers=nn.ModuleList([TransformerBlock(embed_dim,expansion_factor,n_heads) for i in range(num_layers)])#Transformer块
    
    def forward(self,x):#前向传播，x是输入的词索引
        embed_out=self.embedding_layer(x)#词嵌入层
        out=self.positional_encoder(embed_out)#位置编码层
        for layer in self.layers:
            out=layer(out,out,out) #QKV输入时是一个东西
        return out
    
class DecoderBlock(nn.Module):#解码器块
    def __init__(self,embed_dim,expansion_factor=4,n_heads=8):#初始化，embed_dim是词向量维度，expansion_factor是扩展因子，n_heads是多头注意力机制的头部数量
        super(DecoderBlock,self).__init__()#调用父类nn.Module的初始化方法

        self.attention=MultiHeadAttention(embed_dim,n_heads=8)#多头注意力机制
        self.norm=nn.LayerNorm(embed_dim)#层归一化
        self.dropout=nn.Dropout(0.2)#Dropout层
        # print("DecoderBlock初始化",embed_dim,expansion_factor,n_heads)
        self.transformer_block=TransformerBlock(embed_dim,expansion_factor,n_heads)#Transformer块
    
    def forward(self,key,query,x,mask):#前向传播，key,query,x是输入的键、查询和值，mask是掩码
        attention=self.attention(x,x,x,mask=mask)#多头注意力机制
        value=self.dropout(self.norm(attention+x))#Dropout层
        out=self.transformer_block(key,query,value)#Transformer块
        # print("k的值",key.shape)
        # print("q的值",query.shape)
        # print("v的值",value.shape)
        return out
    
    # def forward(self,key,x,value,mask):
    #     attention=self.attention(x,x,x,mask=mask)
    #     query=self.dropout(self.norm(attention+x))
    #     out=self.transformer_block(key,query,value)  
    #     return out

class TransformerDecoder(nn.Module):#Transformer解码器
    def __init__(self, target_vocab_size, embed_dim, seq_len,
                  num_layers=2, expansion_factor=4, n_heads=8):
        #初始化，target_vocab_size是目标词典大小，embed_dim是词向量维度，seq_len是序列长度，
        #num_layers是层数，expansion_factor是扩展因子，n_heads是多头注意力机制的头部数量
        super(TransformerDecoder,self).__init__()#调用父类nn.Module的初始化方法

        self.word_embedding=WordEmbedding(target_vocab_size,embed_dim)#词嵌入层
        self.position_embedding=PositionalEmbedding(seq_len,embed_dim)#位置编码层

        self.layers=nn.ModuleList([DecoderBlock(embed_dim,expansion_factor,n_heads) for _ in range(num_layers)])#解码器块

        self.fc_out=nn.Linear(embed_dim,target_vocab_size)#线性层
        self.dropout=nn.Dropout(0.2)#Dropout层

    def forward(self,x,enc_out,mask):#前向传播，x是输入的词索引，enc_out是编码器的输出，mask是掩码
        x=self.word_embedding(x)#词嵌入层
        x=self.position_embedding(x)#位置编码层
        x=self.dropout(x)#Dropout层

        for layer in self.layers:
            x=layer(enc_out,x,enc_out,mask)#解码器块
        # print("x的值",x.shape)
        out=F.softmax(self.fc_out(x))#softmax层
        return out

class Transformer(nn.Module):#Transformer模型
    def __init__(self,embed_dim,src_vocab_size,target_vocab_size,seq_length,num_layers=2,expansion_factor=4,n_heads=8):#初始化，embed_dim是词向量维度，src_vocab_size是源词典大小，target_vocab_size是目标词典大小，seq_length是序列长度，num_layers是层数，expansion_factor是扩展因子，n_heads是多头注意力机制的头部数量
        super(Transformer,self).__init__()#调用父类nn.Module的初始化方法

        self.target_vocab_size=target_vocab_size#目标词典大小
        self.encoder=TransformerEncoder(seq_length,src_vocab_size,embed_dim,num_layers=num_layers,expansion_factor=expansion_factor,n_heads=n_heads)#编码器
        self.decoder=TransformerDecoder(target_vocab_size,embed_dim,seq_length,num_layers=num_layers,expansion_factor=expansion_factor,n_heads=n_heads)#解码器

    def mask_trg_mask(self,trg):#掩码
        batch_size,trg_len=trg.shape#trg.shape返回张量的形状，batch_size是批量大小，trg_len是目标序列长度

        trg_mask=torch.tril(torch.ones((trg_len,trg_len))).expand(batch_size,1,trg_len,trg_len) 
#tril为下三角函数，torch.ones((trg_len,trg_len))创建一个大小为trg_len*trg_len的单位矩阵，   
# expand(batch_size,1,trg_len,trg_len)将单位矩阵扩展为batch_size*1*trg_len*trg_len的矩阵 
# 在Transformer模型中，目标序列的掩码（target mask）用于防止模型在解码时看到未来的信息。
        return trg_mask
    
    def decode(self,src,trg):#解码，src是源序列，trg是目标序列
        trg_mask=self.mask_trg_mask(trg)#掩码
        enc_out=self.encoder(src)#编码器
        out_lables=[]#输出标签
        batch_size,seq_len=src.shape[0],src.shape[1]#src.shape[0]返回张量的第0维的大小，src.shape[1]返回张量的第1维的大小
        out=trg#目标序列

        for i in  range(seq_len):
            out=self.decoder(out,enc_out,trg_mask)#解码器
            out=out[:,-1,:]  # 提取输出序列中最后一个词元的所有特征
            out=out.argmax(-1) #计算最后一个维度上的最大值索引
            out_lables.append(out.item())#将最后一个维度上的最大值索引添加到输出标签中
            out=torch.unsqueeze(out,axis=0)
        
        return out_lables

    def forward(self,src,trg):#前向传播，src是源序列，trg是目标序列 
        trg_mask=self.mask_trg_mask(trg)#掩码
        enc_out=self.encoder(src)#编码器
        outputs=self.decoder(trg,enc_out,trg_mask)#解码器

        return outputs
    
if __name__ == '__main__':
    src_vocab_size = 11#源词典大小
    target_vocab_size = 11#目标词典大小
    num_layers = 6#层数
    seq_length = 12#序列长度

    # let 0 be sos(start operation signal) token and 1 be eos(end operation signal) token
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
    out = model.decode(src, trg)#解码
    print("打印输出", out)#打印输出