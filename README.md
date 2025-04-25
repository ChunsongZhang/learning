# learning


# llama.cpp的学习笔记

## 快速开始；安装

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build -j

### 1. 设置代理（如需要）
###2. 下载原始模型（以TinyLlama为例）
mkdir -p llama2_models && cd llama2_models
### 3. 转换为GGUF格式
cd ../llama.cpp
python convert_hf_to_gguf.py --outfile models/tinyllama-1.1b-chat-v1.0.f16.gguf --outtype f16 --model-name "TinyLlama-1.1B-Chat" ../llama2_models/tinyllama-original
### 4. 量化模型
./build/bin/llama-quantize models/tinyllama-1.1b-chat-v1.0.f16.gguf models/tinyllama-1.1b-chat-v1.0.q4_0.gguf Q4_0
### 5. 运行量化后的模型
./build/bin/llama-cli -m models/tinyllama-1.1b-chat-v1.0.q4_0.gguf -p "你好，请介绍一下自己" -n 256
### 6. 测试运行：
My journey into education began as an aspiring teacher in college. I majored in Early Childhood Education, and I was inspired by the joy and excitement that young learners bring to their classrooms. I loved the idea of helping children grow and learn, but I also knew that I wanted to make a difference in their lives.
After college, I took a job as a preschool teacher, where I learned the foundational skills needed to become a teacher. I loved the way that children learned through play, and I was inspired by their innate curiosity and desire for knowledge. I went on to receive my master's degree in Early Childhood Education, and then I earned my teaching certification.
As a teacher, I was able to see firsthand the positive impact that I had on my students' lives. They would come to me with questions and needs, and I was able to help them learn through various methods

## Metal Build
On MacOS, Metal is enabled by default.

## llama.cpp 介绍
它把原本只能在大显卡上跑的 LLaMA，“搬”到了日常设备上运行！！！！
//把 llama.cpp 理解成是让你“轻量级运行 LLaMA 模型的神器”。

##GGML是llama.cpp的底层张量计算库，理解GGML对于深入掌握llama.cpp至关重要。
GGML 的核心非常“纯粹”，基本就是用 C 实现了一套张量计算系统，
所以它的核心数据结构围绕张量（tensor）、计算图、上下文（内存管理）来构建。


### GGML核心概念（一个库，规定了一种范式）
- **ggml_tensor**: 张量表示，存储张量数据及其相关属性
- **ggml_op**: 算子表示，定义了张量计算的操作
- **ggml_graph**: 计算图表示，定义了张量计算的顺序
- **ggml_backend**: 执行计算的后端接口(CPU、GPU、Metal等)
- **ggml_backend_buffer**: 存储张量数据的内存缓冲区
- **ggml_gallocr**: 计算图内存分配器，用于高效内存管理

GGML 是一个用 C 语言编写的高性能张量计算库，专为 在 CPU 上高效运行大型语言模型（如 LLaMA） 而设计。它的目标是：
实现推理时无需 GPU
高效使用内存
可运行在轻量级设备

### GGML 内存管理机制详解
1. 预分配的大块内存
在模型加载时，GGML 会根据模型规模、参数维度等一次性分配一整块大内存，例如 4GB 或 6GB。

这一块内存会作为“主内存池”，后续所有的张量（包括中间计算结果、激活、缓存等）都从这块内存中按需分配，不释放。
2. 按需分配的小内存
GGML 会在运行时动态分配小内存，例如张量的激活、缓存等。
- **内存分离设计**：
  - 张量结构体本身是定长的，不直接存储数据。
  - 实际数据通过`data`指针指向非定长内存块。
  - 这种设计实现了结构与数据的分离，便于内存精确分配。

3. 内存池管理
GGML 采用内存池管理机制，将内存分为三类：
- **静态内存池**：用于存储模型参数、激活、缓存等，大小固定，不释放。
- **动态内存池**：用于存储张量数据，大小可变，释放时会回收到主内存池。
- **临时内存池**：用于临时存储张量数据，大小可变，释放时会释放到系统内存。  

4. 无垃圾回收机制，靠重置上下文
GGML 不会自动释放张量的内存，因为这些张量都是从统一大池子里“划出来”的。
因此要清除所有内存，只能销毁整个上下文 ggml_free(ctx)，或重新初始化一次。

ggml 的核心数据结构：
概念 | 类比 PyTorch | 功能解释
ggml_tensor | torch.Tensor | 存数据 + 描述操作的图节点
ggml_context | CUDA context / memory pool | 内存管理器，分配张量用
ggml_cgraph | computation graph | 描述前向图的所有操作流程
ggml_op | 运算符 | 张量是怎么来的？来自 MatMul 还是 Add？
ggml_compute_params | 调度参数 | 用于图执行时调度线程和内存


### GGML 重要的工作流程


#### ggml_context：上下文初始化：
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
//分配张量空间，管理张量

#### ggml 计算图构建
比如调用 llama_eval() 时，LLaMA 会构造 GGML 的 ggml_cgraph：
通过调用 ggml_mul_mat(), ggml_add(), ggml_soft_max() 等 API 构建张量依赖图
最终通过 ggml_graph_compute() 执行

构建计算图graph有两个步骤：
1. 使用算子函数连接weight参数、创建中间计算节点 
2. 使用ggml_build_forward_expand()函数构建计算图。
//第一步	你在搭积木，每块积木知道它是怎么拼上去的（比如“我是 x × W 乘出来的”）
//第二步	你把这些积木整理出施工顺序，列出“先干啥后干啥”，生成一张流程表（图）

![img](https://pic2.zhimg.com/v2-6652dd3bea046d2b6efcb8802a644ebd_1440w.jpg)
就是从“结果”出发，递归访问依赖、区分叶子和操作节点，并以拓扑顺序构建计算图的过程。
目的：明确描述模型中每一步计算的顺序和依赖关系，从而实现高效的自动求导和推理。

###运算图
llama.cpp运算图(computation graph)构建是通过代码驱动的前向过程，而不是从模型文件中解析得来。
运算图专注于节点是操作本身（op），用于底层执行与优化

#### 运算图的可视化表示

llama.cpp中的运算图实际上是这样的结构：

```
输入层(token_ids)
    |
    v
词嵌入(tok_embeddings)
    |
    v
位置编码(RoPE) 
    |
    v
Transformer层1
    |   \
    |    --> 自注意力子图
    |   /         |
    |   \         v
    |    --> 全连接子图
    |   /
    v
Transformer层2
    |   \
    |    --> 自注意力子图
    |   /         |
    |   \         v
    |    --> 全连接子图
    |   /
    v
... [重复N层]
    |
    v
标准化层(rms_norm)
    |
    v
输出投影(output)
    |
    v
logits输出
```






### 调度器
当计算图（如 PyTorch、TensorFlow、ggml）构建完成之后，为了执行图上的各个操作，需要一个“执行器”来决定：
每个 op（操作）何时执行
哪些 op 可以 并行执行
是否要 等待依赖完成，这个“执行器”就是所谓的 调度器（scheduler）。
//op 是“动作的标签”，算子是“干活的工具”


  ```cpp
  struct ggml_backend_sched {
      bool is_reset;            // 是否被重制
      bool is_alloc;            //  是否进行了图分配内存
  
      int n_backends;                                              //运行时，可用的后端数
      ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];            //后端列表
      ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];   //后端buffer列表
      ggml_gallocr_t galloc;                                       //图内存分配器
  
      // 映射graph中的节点的哈希表
      struct ggml_hash_set  hash_set;
      int                 * hv_tensor_backend_ids; // [hash_set.size]
      struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]
  
      // 节点对应backend索引
      int * node_backend_ids; // [graph_size]
      int * leaf_backend_ids; // [graph_size]
      //上一次运行索引记录
      int * prev_node_backend_ids; // [graph_size]
      int * prev_leaf_backend_ids; // [graph_size]
  
      //cgraph以及split子图分割相关
      // copy of the graph with modified inputs
      struct ggml_cgraph graph;
      // graph splits
      struct ggml_backend_sched_split * splits;
      int n_splits;
      int splits_capacity;
  
      // 用于支持流水pipline并行相关副本id索引
      int n_copies;
      int cur_copy;
  
      //记录子图执行的事件
      ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
      //总graph输入
      struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
      int n_graph_inputs;
      struct ggml_context * ctx;
  
      //用户自定义函数接口
      ggml_backend_sched_eval_callback callback_eval;
      void * callback_eval_user_data;
  
      char * context_buffer;
      size_t context_buffer_size;
  
      int debug;
  };
  ```
在推理过程中，每个张量需要在特定的后端上进行计算。通过 backend_id，可以快速找到张量应该在哪个后端上执行。

## llama.cpp对 llama 的量化：

### 函数名 | 功能

ggml_quantize_q4_0() | 把 float 权重压缩为 Q4_0 格式
ggml_quantize_q5_1() | 压缩为 Q5_1 格式
ggml_quantize_chunk() | 通用入口，支持多种格式的分块量化
//

### 量化方式：
按 chunk（每 32 或 64 个值）为单位处理
计算最大值 / 最小值 → 缩放到 [-1, 1]
将每个 float 映射为一个 int4（或 int5）
存储缩放因子和偏移信息

### 量化过程详解：以Q4_0 量化为例
1. 分块处理（Chunking）
权重数据被分成固定大小的块（chunk），每个块包含 32 个浮点数。​

2. 计算缩放因子（Scale）
对于每个块，计算其绝对值的最大值 max_abs，然后确定缩放因子 scale：
scale = max_abs / 7.0f;
（对称量化（Q4_0））这里的 7.0f 是因为 4 位量化可以表示的最大整数值为 7。
3. 量化每个值
对于块中的每个浮点数 x，进行如下操作：
q = round(x / scale);
q = clamp(q, -8, 7); // 限制在 [-8, 7] 范围内
4. 存储量化结果
将每两个 4 位整数打包成一个字节，并存储到目标数组中。同时，记录每个块的缩放因子，以便在反量化时使用。​

#### 量化和类型系统
##### 数据结构
###### 数据类型定义
GGML支持多种数据类型，包括浮点和量化类型：
```c
enum ggml_type {
    GGML_TYPE_F32  = 0,    // 32位浮点
    GGML_TYPE_F16  = 1,    // 16位浮点
    GGML_TYPE_Q4_0 = 2,    // 4位量化（版本0）
    GGML_TYPE_Q4_1 = 3,    // 4位量化（版本1）
    // ...
    GGML_TYPE_Q8_0 = 8,    // 8位量化（版本0）
    GGML_TYPE_Q8_1 = 9,    // 8位量化（版本1）
    // k量化系列
    GGML_TYPE_Q2_K = 10,   // 2位K量化
    GGML_TYPE_Q3_K = 11,   // 3位K量化
    // ...其他类型
};
```

###### 类型特性与量化
- **类型特性**：每种类型都有对应的`ggml_type_traits_t`结构，包含：
  ```c
  typedef struct {
      const char * type_name;          // 类型名称
      int blck_size;                   // 块大小
      size_t type_size;                // 类型大小
      bool is_quantized;               // 是否量化
      ggml_to_float_t to_float;        // 转换为浮点函数
      ggml_from_float_t from_float;    // 从浮点转换函数
      ggml_from_float_t from_float_reference; // 参考转换函数
      ggml_vec_dot_t vec_dot;          // 向量点积函数
      enum ggml_type vec_dot_type;     // 向量点积类型
  } ggml_type_traits_t;
  ```

- **量化函数**：提供了多种量化函数如`ggml_quantize_q4_0`、`ggml_quantize_q4_1`等，用于将浮点数据转换为低精度表示。


llama.cpp中涉及到量化推理的主要就是Linear层，前文提过本文先导知识之一就是模型量化，所谓模型量化就是将模型中的weight数据和input-tensor数据，通过量化算法将原始FP32类型逻辑等价地转换为int8以及更低bit数据，这样做的好处就是在对模型进行推理时能节省内存和计算加速的好处。模型量化算法有很多种，以常见的对称均匀量化为例，模型量化时都会对原始FP32数据在pre-tensor/pre-channel域计算得到一个scale，然后通过量化公式：q =round(clip(r_i /scale,Q_{min},Q_{max}))将数据由FP32量化为INT-8(或更低bit)数据 。

这里解释一下：模型量化后计算速度的加快的主要原因在于：在同等带宽的情况下能一次向量化的load更多数据(比如原始load 1个FP32的时间 现在能load 4个int8的数据)
以llama.cpp 提供的LLaMA 2 7B chat 8bit模型为例，Llama 2中Linear层的weight数据就是int-8类型，更具体的说，Linear层中的weight数据是以如下结构体的形式保存的，其中d为前文中提到的量化算法中的scale，int8_t qs[QK8_0]即为量化后的INT-8数据
```c
#define QK8_0 32
typedef struct {
    half    d;              // delta 量化的scale
    int8_t  qs[QK8_0];      // quants 量化的weight数据
} block_q8_0;
```


## 反量化（dequantization）
 是将量化后的整数值（如 int8）乘以一个缩放因子（scale），还原出近似原始的浮点数值。

方法一（反量化权重）优势：
计算精度高：FP32计算保留了更多数值精度，减少量化误差传播
激活函数兼容性好：FP32结果可直接用于各种激活函数
实现简单：不需要为输入设计专门的量化策略
梯度计算友好：如果需要微调，FP32更适合反向传播
方法一劣势：
内存带宽消耗大：需要额外的反量化操作
计算速度较慢：FP32计算比INT8计算需要更多资源
缓存效率低：反量化后的权重占用更多缓存空间
方法二（量化输入）优势：
计算速度快：INT8矩阵乘法通常比FP32快2-4倍
内存带宽需求低：数据传输量显著减少
能耗效率高：INT8运算比FP32运算更节能
硬件加速：现代CPU/GPU对INT8运算有专门优化(SIMD指令)
方法二劣势：
精度损失较大：两次量化(权重和输入)导致误差累积
动态范围受限：INT8表示范围有限，可能导致溢出
实现复杂：需要为输入设计量化方案并处理边界情况
量化校准复杂：输入量化参数需要根据不同数据调整

## numa
### NUMA优化

对于多处理器系统，NUMA优化可以提高性能:

```bash
# 指定NUMA节点
./llama-cli -m models/your-model.gguf --numa 0
```
NUMA（非统一内存访问架构）是一种适用于多处理器系统的内存架构设计，在该架构中：
基本概念
内存访问不均匀：处理器访问自己本地节点的内存比访问远程节点的内存更快
节点划分：系统被划分为多个"NUMA节点"，每个节点包含自己的处理器和本地内存
节点间通信：节点之间通过互连总线通信，访问远程内存比本地内存慢2-10倍
NUMA优化目标
在llama.cpp中进行NUMA优化是为了：
确保计算和内存分配尽可能地在相同NUMA节点内完成
减少跨NUMA节点的内存访问
最小化处理器等待远程内存的时间
实际优化方式
llama.cpp的NUMA优化主要通过以下方式实现：
内存分配策略：尽量在处理器本地节点分配内存
线程绑定：将计算线程绑定到特定的CPU核心和NUMA节点
数据本地化：确保每个线程处理的数据尽可能位于同一NUMA节点
应用场景
NUMA优化主要在以下环境中显著提升性能：
多插槽服务器（多个物理CPU）
高性能计算集群
AMD ThreadRipper/EPYC或Intel Xeon等多NUMA节点的处理器
单节点系统（如普通台式机、笔记本电脑）通常不需要NUMA优化，因为它们只有一个NUMA节点。


### NVIDIA专用
- **CUDA图优化**：`-DGGML_CUDA_GRAPHS=ON` 用于连续批处理时
- **精度控制**：`-DGGML_CUDA_F16=ON` 对于精度要求较低场景

### Apple专用
- **Metal精度控制**：`-DGGML_METAL_USE_BF16=ON` 适用于高性能场景，可能导致精度降低。m4性能提升明显。
llama.cpp运用了运行时量化计算的方法，开启该功能后部分量化计算会被跳过，在极端情况下可能会重叠导致进一步放大误差。
这些调优技巧应根据您的特定硬件和用例进行调整，以获得最佳性能。


##### SIMD优化与硬件加速

###### 1. SIMD映射
GGML定义了一组通用的C宏，映射到特定架构的SIMD指令：
```c
// NEON指令集优化
#define GGML_F32x4              float32x4_t
#define GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#define GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#define GGML_F32x4_LOAD         vld1q_f32
#define GGML_F32x4_STORE        vst1q_f32
#define GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
// ...更多指令
```

###### 2. 硬件特性检测
- 提供了一系列函数检测CPU特性：
  ```c
  int ggml_cpu_has_avx(void);
  int ggml_cpu_has_avx2(void);
  int ggml_cpu_has_avx512(void);
  int ggml_cpu_has_fma(void);
  int ggml_cpu_has_neon(void);
  int ggml_cpu_has_arm_fma(void);
  int ggml_cpu_has_f16c(void);
  // ...更多检测函数
  ```

##### 七、后端与扩展系统

###### 1. 后端类型
- **基础后端**：
  ```c
  enum ggml_backend_type {
      GGML_BACKEND_CPU = 0,    // CPU后端
      GGML_BACKEND_GPU = 10,   // GPU后端
      GGML_BACKEND_GPU_SPLIT = 20, // 分离GPU后端
  };
  ```

###### 2. 缓冲区管理
- **后端缓冲区**：通过`ggml_backend_buffer`管理不同后端的内存
- **缓冲区类型**：使用`ggml_backend_buffer_type`定义不同类型的缓冲区


### Split 子图分割
（Graph Partitioning / Subgraph Split） 是指：
把一个完整的计算图（Computation Graph）按照某种规则拆成多个子图（subgraphs），每个子图可以单独执行、部署或优化。

#### def：
每个张量单独分配后端的目的：提高并行度 and 减少内存占用
扩展绑定：在计算图或张量系统中，将额外的资源（如后端设备、执行逻辑、缓存策略等）“绑定”到张量或计算节点上的机制。


#### 步骤： 

1. 遍历cgraph中的所有tensor，使用sched中的哈希映射结构来配对保存每个tensor和对应的backend_id（跳过view_tensor）
2. 扩展绑定上一个步骤中，因未分配到合适的backend_id的tensor（跳过view_tensor）
3. 检查是否有可以进行backend后端升级的tensor（例如某个tensor既可以在cpu中也可以在gpu中时，便将其升级的gpu中）
4. 根据dst和view_src分配剩余未分配的的src节点（view_tensor等）
5. 进行split子图分割


### 根据backend_id进行子图分割（如果 splits 的 0,1 节点不在同一个backend_id，再建一个节点存在哈希表里对接）

由于之前的过程，我们可以知道在后续计算过程中，我们只需要遍历所有的nodes节点就可以完成整个网络的计算。所以对应此时的子图分割，我们也只需要遍历nodes节点即可。而分割的原则也很简单，那就是从0-n进行遍历，在所有bakend_id变化之处（如CPU->GPU或GPU->CPU）进行切割，形成一个子图：“**ggml_backend_sched_split \* split**”。如下图所示：

![img](https://pica.zhimg.com/v2-b0525eb465ca28534ef8fc71732affae_1440w.jpg)

但是此时，我们还不能进行compute计算，因为得到的sched->splits*子图是分离的状态，实际上sched->splits[1]的输入需要依赖于sched->splits[0]的输出，并且这两个子图不在同一个backend后端设备上，还需要涉及到跨后端传输的问题。所以接下来我们还需要对分割后的子图进行输入input设置，将他们连起来。
首先在split子图分割时，会同时记录分割后子图的前一个node节点，将其当作输入input，记录于splits[i]->input*指针。并且还会检查每个splits[i]->input所在的backend后端与当前整体子图的backend后端是否为同一个。在本例程中，如下图所示，第一个子图的splits[0]->input与split[0]子图都处于CPU后端，而第二个子图的splits[1]->input位于CPU后端，但是splits[1]子图本身位于GPU后端。

![img](https://picx.zhimg.com/v2-c31881ea715b195650f4674963a260ab_1440w.jpg)
当发生对应的input与splits后端不一致时，会使用ggml_dup_tensor_layout()函数在当前子图：splits[1]中添加一个孪生tensor节点，其内存布局与splits[1]->input*指针指向的tensor完全相同，只是对应的backend不同。并且使用tensor_id_copy()哈希函数进行记录，方便后续compute时进行快速查找。
至此，我们就完成了从cgraph静态线性图到sched->splits子图的转换过程。

### 实例：
##### 1. 模型加载
- **文件映射**：使用`gguf_init_from_file`将GGUF文件映射到内存，提高加载效率
- **模型解析**：创建专用`ggml_context`管理模型张量内存，解析GGUF结构
- **内存映射优势**：
  - 避免完整加载大型模型
  - 操作系统可智能管理内存页
  - 支持多进程共享

##### 2. 计算内存管理
- **上下文初始化**：通过`ggml_init()`创建上下文并分配固定大小内存
- **内存布局**：
  - 上下文内存被分为一块连续区域
  - 张量按顺序在此区域中分配
  - 不支持动态增长，必须预先分配足够空间
- **张量分配**：使用`ggml_new_tensor`系列函数在上下文中创建张量
- **中间结果**：每个操作创建新节点存储中间结果，插入同一上下文

##### 3. 额外空间管理
- **计算计划**：`ggml_cplan`结构管理计算执行需要的临时内存
  ```c
  struct ggml_cplan {
      size_t work_size;    // 所需工作空间大小
      uint8_t * work_data; // 工作空间指针
  };
  ```
- **用途**：
  - 矩阵乘法中间结果
  - 特定算子的临时缓冲区
  - 并行计算的共享数据
- **分配方式**：可静态分配或动态申请
  ```c
  ggml_cplan cplan = ggml_graph_plan(gf, n_threads);
  uint8_t * work_data = malloc(cplan.work_size);
  cplan.work_data = work_data;
  ```

######  技术特点
- **预分配内存**：通过context预分配内存，减少运行时内存分配开销
- **链表管理**：使用链表管理对象和内存块
- **精确内存分配**：tensor结构体内部不存放真实数据，通过指针实现数据分离
- **计算图优化**：通过node和leaf概念构建高效计算图

##### 核心数据结构详解

###### 1. Context结构
GGML的核心是`ggml_context`结构，它作为整个系统的内存管理器：

```cpp
struct ggml_context {
    size_t mem_size;            // 内存总大小
    void * mem_buffer;          // 内存缓冲区指针
    bool   mem_buffer_owned;    // 内存缓冲区是否由context拥有
    bool   no_alloc;            // 是否禁止分配
    bool   no_alloc_save;       // 保存no_alloc状态，用于临时scratch缓冲区

    int    n_objects;           // 对象数量

    struct ggml_object * objects_begin;  // 对象链表头
    struct ggml_object * objects_end;    // 对象链表尾

    struct ggml_scratch scratch;         // 临时计算缓冲区
    struct ggml_scratch scratch_save;    // 保存的临时计算缓冲区
};
```

这个结构负责:
- 管理所有内存分配
- 跟踪所有创建的对象
- 维护对象链表
- 管理临时计算缓冲区

###### 2. Object对象
对象是Context中的基本管理单元：

```cpp
struct ggml_object {
    size_t offs;                // 内存偏移量
    size_t size;                // 对象大小
    struct ggml_object * next;  // 链表下一个元素
    enum ggml_object_type type; // 对象类型 (TENSOR, GRAPH等)
    char padding[4];            // 填充对齐
};
```

每个对象通过链表连接，记录了在内存池中的偏移量和大小。

###### 3. Tensor张量
张量是GGML中最重要的数据结构：

```cpp
struct ggml_tensor {
    enum ggml_type         type;    // 数据类型 (F32, F16, Q4_0等)
    enum ggml_backend_type backend; // 后端类型 (CPU, GPU等)
    struct ggml_backend_buffer * buffer; // 后端缓冲区

    int64_t ne[GGML_MAX_DIMS]; // 每个维度的元素数
    size_t  nb[GGML_MAX_DIMS]; // 字节步长
    
    enum ggml_op op;           // 操作类型
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]; // 操作参数
    
    bool is_param;             // 是否为参数
    
    struct ggml_tensor * grad; // 梯度张量
    struct ggml_tensor * src[GGML_MAX_SRC]; // 源张量
    
    // 性能统计
    int     perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;
    
    struct ggml_tensor * view_src; // 视图源
    size_t               view_offs; // 视图偏移
    
    void * data;               // 实际数据指针
    
    char name[GGML_MAX_NAME];  // 张量名称
    
    void * extra;              // 额外数据（如CUDA相关）
    
    char padding[8];           // 填充对齐
};
```

张量结构包含：
- 类型和形状信息
- 操作和参数
- 数据指针
- 计算图连接（src和grad）
- 性能统计信息

内存池(mem_buffer)
+----------------+
| ggml_object 1  | ← objects_begin
+----------------+
| ggml_tensor 1  | ← 第一个张量
+----------------+
| tensor 1 data  | ← 如果分配在context中
+----------------+
| ggml_object 2  |
+----------------+
| ggml_tensor 2  | ← 第二个张量
+----------------+
| tensor 2 data  | ← 如果分配在context中
+----------------+
|      ...       |
+----------------+
| ggml_object N  | ← objects_end
+----------------+
| ggml_tensor N  | ← 最后一个张量
+----------------+
| tensor N data  | ← 如果分配在context中
+----------------+
```

###### 2. Data与Context的关系

张量的数据(data)可以存储在以下几个位置：

1. **Context内存池中**：
   - 当张量不是view且没有设置no_alloc，且没有scratch缓冲区时
   - 数据存储在张量结构体后面：`data = (void *)(tensor + 1)`
   - 连续存储，管理简单

2. **Scratch缓冲区中**：
   - 用于临时计算结果
   - 当存在scratch缓冲区且创建张量时选择使用它
   - 数据存储在`scratch.data + scratch.offs`

3. **Backend Buffer中**：
   - 当使用GPU等后端时
   - 数据存储在特定后端的缓冲区中
   - `buffer`字段指向后端缓冲区

4. **外部内存**：
   - 当张量是view或设置了no_alloc
   - 数据指针指向外部提供的内存

###### 3. 数据分配的核心逻辑

张量数据分配的关键代码（简化版）：

```cpp
void * data = view_src != NULL ? view_src->data : NULL;
if (data != NULL) {
    data = (char *) data + view_offs;
}

size_t obj_alloc_size = 0;

if (view_src == NULL && !ctx->no_alloc) {
    if (ctx->scratch.data != NULL) {
        // 在scratch缓冲区分配
        data = ctx->scratch.data + ctx->scratch.offs;
        ctx->scratch.offs += data_size;
    } else {
        // 在context内存池中分配
        obj_alloc_size = data_size;
    }
}

// 创建张量结构体
struct ggml_tensor * result = (struct ggml_tensor *)(ctx->mem_buffer + obj_new->offs);

// 设置data指针
result->data = obj_alloc_size > 0 ? (void *)(result + 1) : data;
```

### Llama.cpp内存管理机制与核心架构解析

#### 一、系统架构与核心模块

##### 1. 层次化模块设计
Llama.cpp采用了高度模块化的结构，主要包括以下几个核心组件：

- **模型层（Model）**：包含模型权重、结构定义和参数配置，定义数据结构
- **上下文层（Context）**：管理推理状态、KV缓存和计算资源，分配资源
- **计算图层（Graph）**：构建和优化模型计算流程，计算中间节点
- **内存管理层（Memory）**：高效管理模型权重和中间计算结果，使用大块内存，管理权重和中间向量
- **KV缓存层（KV Cache）**：特化的内存管理系统，用于存储注意力机制中的键值对，维护所有的 kvtoken
- **批处理层（Batch）**：优化多序列的并行处理，支持多输入、多输出的模型推理












# 微调：
LoRA：LoRA 是一种“低秩权重插入微调”方法，它在不改动原始模型权重的情况下，通过训练“附加的少量参数”来完成模型微调。


## LoRA 的核心思想
假设我们在做全连接层的微调（如下）：
原始：y = W x
我们不更新 W，而是：
LoRA：y = (W + ΔW) x，其中 ΔW ≈ A B （低秩分解）
W：原始冻结权重（不更新）
ΔW = A × B：LoRA 引入的可训练部分
A: 小矩阵，shape 是 r × d
B: 小矩阵，shape 是 d × r
通常 r ≪ d，比如 r=8, d=4096，所以新增参数极少
y=Wx+A(Bx)
这样：原模型权重不变
引入的 A, B 参数非常少（百万级别而不是几十亿）
整体计算成本、显存、训练时间大幅下降

#此外，Transformer的权重矩阵包括Attention模块里用于计算query, key, value的Wq，Wk，Wv
以及多头attention的Wo,以及MLP层的权重矩阵，LoRA只应用于Attention模块中的4种权重矩阵，
而且通过消融实验发现同时调整 Wq 和 Wv 会产生最佳结果。
（在 Transformer 中，“隐藏层”通常指的是每一个 Encoder Block / Decoder Block 内部的中间处理层（尤其是多头注意力 + MLP 部分）。）


## lora 运行测试输出：
标准模型准确率: 0.5210
LoRA模型准确率: 0.2540
LoRA测试完成！
结论：
1. LoRA模型可训练参数减少了 86.62%
2. LoRA模型与标准模型的准确率：0.2540 vs 0.5210
3. LoRA适合于资源受限的环境，特别是在微调大型预训练模型时，LoRA的参数效率更高，但是在小模型上，LoRA的性能表现较差

## AdaLoRA
SVD 是把一个任意矩阵拆分成三个部分：方向 × 拉伸 × 方向。
A = U · Σ · Vᵀ
其中 U, V 是正交矩阵，Σ 是对角矩阵，U 的列向量是左奇异向量，V 的列向量是右奇异向量。

AdaLoRA 就是用 SVD 去掉 A 的某些奇异值，然后用这些奇异值来近似 A。

### adalora 流程
行数 | 中文解释 | 实际作用
第1行 | 设置超参数（训练轮次、rank预算、学习率等） | 初始化
第2-9行 | 开始训练循环 | 整体过程控制
第3行 | 计算损失对参数的梯度 | 获取方向信息
//（从数据集中采样 batch，计算当前 LoRA 参数 P,E,Q 的梯度）
第4行 | 计算梯度的敏感性 | 衡量参数变动对损失的影响
//计算每个 LoRA 参数的敏感性指标 I （t）
第5行 | 计算“滑动值”——用来平滑敏感性指标 | 抗噪声（估计 u 和i）
第6行 | 估计每个参数组的“重要性” | 核心！用来决定谁配更多预算
//计算每个参数子块 k、秩位置 i 的重要性分数
第7行 | 根据梯度更新主 LoRA 参数 PPP、QQQ | 训练主体//梯度下降
第8行 | 根据预算修剪更新 Λ | 动态控制 rank 的策略实现//对 Λ（控制秩的门控变量）也进行梯度更新

模块 | 作用
敏感性计算 I(t)I^{(t)}I(t) | 衡量哪个参数对 loss 敏感
不确定性 Uˉ(t)\bar{U}^{(t)}Uˉ(t) | 抗梯度噪声，防止误删重要方向
剪枝策略 T\mathcal{T}T | 控制秩动态变化（谁被裁，谁保留）
Λ 控制门（门控） | 控制秩的精细开关，类似 attention mask
预算 b(t)b^{(t)}b(t) | 在训练过程中逐步调整允许的秩总数（参数预算）

//LoRA 和 AdaLoRA 的本质都是： 👉 在不改变原始大模型的前提下，只引入少量低秩参数来微调模型表现。


## AraLora运行测试输出：
AdaLoRA优势测试完成！
结论：
1. 初始时，AdaLoRA和LoRA模型的参数数量相同，都减少了 98.34% 的参数
2. 训练结束后，AdaLoRA通过自适应调整进一步减少了 50.04% 的参数
3. 尽管参数更少，AdaLoRA模型的性能与标准模型和LoRA模型相当（准确率：0.1170 vs 0.1050 vs 0.1590）
4. AdaLoRA通过重要性评分动态分配参数预算，保留最重要的低秩成分
5. 随着训练的进行，AdaLoRA能够逐渐减少参数数量，同时保持模型性能
6. AdaLoRA特别适合于资源受限的环境，可以在保持性能的同时进一步减少训练参数

## QLoRA
 QLoRA = Quantization + LoRA

 组成 | 含义
Quantization | 把模型参数压缩为 4bit（如 NF4 格式），大幅减少显存
LoRA | 冻结主模型，只在低秩插入点训练少量参数
Double Quantization | 更进一步压缩存储空间（用 8bit 存 scale）
Paged Optimizers | 降低优化器状态开销（用于 GPU VRAM 不够的场景）

##  Prefix Tuning

  前缀微调（Prefix-Tuning）是一种轻量级的微调方法，受提示（Prompting）的启发，它引入了可训练的连续前缀向量，作为任务特定的参数。该方法通过在输入序列前添加一组可训练的前缀向量（prefix），模型在生成时可以将其视为“虚拟的”提示，使得预训练语言模型能够在不修改其原有参数的情况下，适应特定任务。这些前缀向量在模型的每一层中都存在，作为额外的上下文信息引导模型生成符合任务需求的输出。


##  Prefix Tuning
  - **前缀注入**：
  为每一层 $l$，引入前缀向量 $P^l = [p_1^l, p_2^l, \dots, p_L^l]$，其中 $L$ 是前缀长度。

  - **自注意力机制调整**：
  在第 $l$ 层的自注意力计算中：
   
    $Q = X W_Q$

    $K = [P^l; X] W_K$

    $V = [P^l; X] W_V$


## Prompt Tuning

“不改模型，不插模块，也不改 attention，只学输入前几个 token 向量。”
不适合复杂任务（如结构化输出），但非常适合大模型指令微调，Prompt Tuning 是参数量最少、结构最简单的微调方法，适合轻量化场景，通过学习一段“软提示”引导语言模型完成新任务。
**例子讲解**

例1：想象你是一家通用快递公司的快递员，你能够处理各种包裹。然而，现在你有一些特殊的包裹需要额外处理，比如易碎品。在 Prompt Tuning 中，相当于你在每个易碎品包裹前面加上一个特别的标签，比如“易碎”，以提醒你在处理这些包裹时要更加小心。例如，原始包裹信息是：“Package contains glassware”（包裹内含玻璃制品）。在 Prompt Tuning 中，你会在包裹信息前加上“易碎”标签，让信息变成：“Fragile: Package contains glassware”。这些“Fragile”标签是可训练的，通过训练这些标签，你可以更好地处理易碎品。

例2：假设你有一个已经训练好的模型，能够回答通用问题。现在你希望它能够更好地回答旅游相关的问题。原始输入句子是：“What is the best place to visit in summer?”（夏天最好的旅游地点是哪里？）。在 Prompt Tuning 中，你会在输入句子前添加一些额外的 Token，比如 [TRAVEL]，让输入变成：[TRAVEL] What is the best place to visit in summer? 这些 [TRAVEL] Token 是可训练的，通过训练这些 Token，你可以让模型更好地理解这是一个关于旅游的问题。

**原理解释**

`Model Tuning` 需要为每个下游任务创建一个特定任务的完整预训练模型的副本，并且推理必须在单独的批次中执行。

`Prompt tuning` 仅需要为每个任务存储一个小的特定任务提示，并且可以使用原始的预训练模型进行多任务推理。

对于一个 T5 “XXL” 模型，每个经过微调的模型副本需要 110 亿个参数。相比之下，如果采用提示微调（Prompt Tuning），每个任务仅需存储 20,480 个参数，相比完整微调，参数量减少了五个数量级，假设提示的长度为 5 个 token。

## Prompt Tuning测试运行
结论：
1. Prompt-Tuning模型可训练参数减少了 99.82%
2. 尽管参数更少，Prompt-Tuning模型的性能与标准模型相当（准确率：0.1210 vs 1.0000）
3. Prompt-Tuning通过添加少量可学习的连续向量，有效地调整模型行为
4. Prompt-Tuning特别适合于资源受限的环境，可以在保持性能的同时大幅减少训练参数
5. 提示长度是一个重要的超参数，影响模型性能和参数效率
6. 与P-Tuning和Prefix-Tuning相比，Prompt-Tuning结构更简单，不需要额外的编码器网络
7. Prompt-Tuning适用于各种NLP任务，特别是在大型预训练模型上进行任务适应时


## P-Tuning（Prompt Tuning v2） 
一种用可学习的连续 embedding 向量作为“软提示”插入模型输入中，同时使用**深层嵌入策略（如 LSTM）**增强表示力的参数高效微调方法。

它是 Prompt Tuning 的进阶版。
目标是：保留轻量微调的优点，同时显著提升表达能力 ✅

P-Tuning，该方法将Prompt转换为可以学习的Embedding层，并用MLP+LSTM的方式来对Prompt Embedding进行一层处理。


## **微调数据集介绍**

**预训练**（Pre-training）的过程中，我们一般用**海量非结构化文本**（比如书籍、网页、对话），通过「预测下一个词」来训练模型，这也就意味着预训练的数据集格式是没有明确要求的。

**监督微调**（Supervised Fine-Tuning，SFT），顾名思义就是需要人去监督微调的过程，需要明确这是什么问题，正确答案是什么。

- **指令微调**

  如果我们想让模型具备多种语言理解的能力，这时候只靠两个字段就不够了，因为在 `Input` 是同样一个词语的时候，根据我们想让模型完成的不同任务，`output` 可能是不一样的，这时候我们就要多引入一个指令的概念，比如这个数据集：

  ```json
  [
    {
      "instruction": "将这句英文翻译成法语",
      "input": "Hello, how are you?",
      "output": "Bonjour, comment ça va ?"
    },
  ]
  ```
  

  我们告诉模型明确的指令：将英文翻译为法语，再将 `Input`（英文）、`Output`（法语）告诉模型， 模型就能准确理解要做什么了，这就是指令微调。

  [指令微调典型开源数据集](https://huggingface.co/datasets/shibing624/alpaca-zh)

- **对话微调**

  对话微调（`Dialogue Tuning`） 是通过多轮对话数据训练模型生成连贯、符合语境的回复，强调对话历史的上下文理解和回复的自然流畅性。其核心在于教会模型处理对话中的逻辑关系、情感表达和角色身份，对话微调的数据集通常包含对话的上下文以及对应的回复

  ```json
  [
    {
      "dialogue": [
        {"role": "user", "content": "今天天气怎么样？"},
        {"role": "assistant", "content": "北京今日多云转晴，气温22℃，适合户外活动。"},
        {"role": "user", "content": "那适合去长城吗？"},
        {"role": "assistant", "content": "长城景区海拔较高，建议携带外套，注意防晒。"}
      ]
    },
  ]
  ```

  [对话微调典型开源数据集](https://huggingface.co/datasets/philschmid/guanaco-sharegpt-style)

- **领域适配**

  领域适配（`Domain Adaptation`）是指将模型在特定领域的数据上进行微调，使其更好地适应特定领域的任务和需求。这些数据集通常包含该领域的专业术语、特定格式和相关任务的标注。例如，在医疗领域，数据集可能包含病历文本、医学术语以及对应的诊断结果等信息。

  ```json
  [
    {
      "instruction": "分析患者的症状描述",
      "input": "55岁男性，持续性胸骨后疼痛3小时，含服硝酸甘油无效",
      "output": "可能诊断：急性心肌梗死（STEMI），建议立即行心电图检查及心肌酶谱检测",
      "domain": "医疗"
    },
    {
      "instruction": "解释法律条款",
      "input": "《民法典》第1032条",
      "output": "该条款规定自然人享有隐私权，任何组织或个人不得以刺探、侵扰、泄露、公开等方式侵害他人隐私权",
      "domain": "法律"
    },
  ]
  ```

  [领域适配典型开源数据集](https://huggingface.co/datasets/qiaojin/PubMedQA)

- **文本分类**

  文本分类（`Text Classification`），它是自然语言处理中的一个经典任务，目的就是通过标注数据训练模型对文本进行类别预测或标签分配。我们需要使用标注了类别的文本数据集对模型进行训练，让模型学习文本特征与类别的映射关系。文本分类数据集的关键在于构建符合业务需求的分类标签，例如从评论中区分出好评和差评，从新闻中区分出客集新闻和金融新闻。

  ```json
  [
    {"text": "这款手机续航长达48小时，拍照效果惊艳", "label": "positive"},
    {"text": "系统频繁卡顿，客服响应速度慢", "label": "negative"},
    {"text": "量子计算机突破新型纠错码技术", "label": "science_news"},
    {"text": "央行宣布下调存款准备金率0.5个百分点", "label": "finance_news"}
  ]
  ```

  [文本分类典型开源数据集](https://huggingface.co/datasets/stanfordnlp/imdb)

- **模型推理微调**

  推理模型的微调其实是监督微调的一种特殊形式，通过在数据集中显式标注思维链（`Chain of Thought, COT`），训练模型不仅给出最终答案，还能生成逻辑推导过程。其核心在于让模型学会「分步思考」，适用于需要复杂逻辑推理的场景（如数学证明、代码调试）。

  在推理模型（比如 `DeepSeek-R1`）的回答中，`<think></think>` 中包含的这部分其实就是模型的推理过程，它其实是根后面的答案一起作为一个回答输出的，只不过在大部分的 C 端应用中对这部分提取出来做了特殊展示。

  ```json
  [
    {
      "instruction": "解决数学应用题",
      "input": "小明买了3支铅笔，每支2元；又买了5本笔记本，每本比铅笔贵4元。总花费多少？",
      "chain_of_thought": [
        "铅笔单价：2元/支 → 3支总价：3×2=6元",
        "笔记本单价：2+4=6元/本 → 5本总价：5×6=30元",
        "合计花费：6+30=36元"
      ],
      "output": "总花费为36元"
    },
  ]
  ```

  注意：不是所有任务都适合用推理模型。因为推理模型的幻觉比较大，有些情况选择推理模型反而会起到相反的效果，在处理简单明确的任务时，推理模型可能会把问题复杂化，导致思考过度、响应较慢，甚至增加幻觉的风险。比如如果你让推理模型去完成检索、解释类的任务时，当它找不到可以参考的信息就会按照自己的思考过程进行输出，结果并不一定准确

- **强化学习微调**

  强化学习微调（`Reinforcement Learning from Human Feedback，RLHF`）是在监督微调的基础上，通过人类来主动反馈优化模型生成质量的方法。

  其核心在于引入奖励模型（`Reward Model`）评估生成结果的合理性，并通过强化学习策略（如 `PPO` 算法）调整模型参数，使生成内容更符合人类偏好。

  ```json
  [
    {
      "input": "请推荐一部科幻电影",
      "output": "《星际穿越》是一部经典科幻片，探讨了时间与亲情。",
      "reward_score": 4.5
    },
    {
      "input": "解释黑洞理论",
      "output": "黑洞是由暗物质构成的神秘天体，会吞噬一切物质。",
      "reward_score": 2.0
    }
  ]
  ```

  **强化学习微调的典型业务场景**

  - **对话系统优化**：提升回复的相关性，对齐人类价值观（安全、无害、有用性）。
  - **内容生成**：控制输出风格（如幽默、正式）或避免敏感信息。
  - **代码生成**：优化代码的可读性和正确性。

  [强化学习典型开源数据集](https://huggingface.co/datasets/Dahoas/rm-static)

- **多模态微调**

  多模态微调（`Multimodal Fine-Tuning`）指通过文本、图像、语音等多模态数据训练模型，使其具备跨模态理解与生成能力。它和文本类模型的微调可以说是并列的两个范畴，其中也包括监督/非监督微调、强化学习微调等范畴。

微调模型的参数中有两个重要的参数：

- train_datset：接收上面我们已经处理好的数据集
- dataset_text_field：用于指定取数据集中的哪个字段来做训练。


## 混合精度训练：
混合精度训练 是指在模型训练过程中，使用低精度（如 float16）来进行大部分计算，而用高精度（float32）来进行某些关键计算（如梯度更新），从而加速训练并减少内存消耗。

### 实现混合精度训练有两种方式

- **在纯 PyTorch 中手动管理**混合精度及训练循环

  可参考代码`pytorch_mixed_precision_training.py`

  ```python
  # 在混合精度训练中，使用梯度缩放器 (GradScaler) 来防止数值下溢
  scaler = GradScaler()
  
  # 在 autocast 上下文中执行前向与反向计算，并指定 dtype=torch.bfloat16
  with autocast(device_type='cuda', dtype=torch.bfloat16):
      output = model(input_data)
      loss = criterion(output, target)
  
  # 使用梯度缩放器来反向传播并更新参数
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  ```

- **使用 Hugging Face Transformers 框架的高级接口**来启用混合精度训练

  可参考代码`huggingface_api_mixed_precision_training.py`

  `bf16=True` 就是开启 **混合精度训练的关键参数**。Transformers 库会在底层使用 PyTorch 的 AMP 机制，并完成 BF16 autocast 与梯度缩放（GradScaler）的管理。

  ```python
  model = AutoModelForCausalLM.from_pretrained("gpt4")
  
  # 开启混合精度训练
  training_args = TrainingArguments(
      output_dir="./output",
      per_device_train_batch_size=8,
      num_train_epochs=3,
      learning_rate=5e-5,
      bf16=True ,  # ✅ 启用混合精度 也可以使用 fp16=True
      logging_steps=10,
  )
  
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=your_dataset,
  )
  
  trainer.train()
  ```

<br>



### 微调框架介绍

1. **Llama Factory**

    LLaMA-Factory作为一个开源的微调框架，通过其用户友好的界面和丰富的功能特性，为开发者提供了一个简便、高效的工具，以便在现有的预训练模型基础上，快速适应特定任务需求，提升模型表现。

    [官网安装教程](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html#)

    **1.1 LLaMA-Factory配置**

    **安装LLaMA-Factory**

    ```bash
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory

    conda create -n llama-factory python=3.10
    conda activate llama-factory

    pip install -e ".[torch,metrics]"
    ```

    **LLaMA-Factory校验**

    ```bash
    llamafactory-cli version
    ```

    **启动LLaMA-Factory WebUI**

    ```bash
    CUDA_VISIBLE_DEVICES=0 GRADIO_SHARE=1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui
    ```

    **打开WebUI**

    ```bash
    localhost:7860
    ```

    **1.2 下载模型**

    **安装huggingface模型下载工具**

    ```bash
    export HF_ENDPOINT=https://hf-mirror.com

    # 安装模型下载工具
    pip install -U huggingface_hub
    pip install huggingface-cli
    ```

    **模型下载**

    选择[llama3模型](https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat)做为基础模型

    ```bash
    # 下载到本地
    huggingface-cli download --resume-download shenzhi-wang/Llama3-8B-Chinese-Chat --local-dir /root/autodl-tmp/models/Llama3-chinese
    ```



    **1.3 数据集下载**

    选择关于经济学对话信息的[数据集](https://github.com/echonoshy/cgft-llm/blob/master/llama-factory/data/fintech.json)，将文件放置到`LLaMA-Factory/data`目录下。

    ```bash
    git clone https://github.com/echonoshy/cgft-llm.git

    cp cgft-llm/data/fintech.json ~/LLaMA-Factory/data
    ```

    **修改数据注册文件**

    修改文件：`LLaMA-Factory/data/dataset_info.json`，目的是为了让LLaMA-Factory识别到我们的数据集文件。注：其中columns指定了数据集的列。

    ```
    "fintech": {
    "file_name": "fintech.json",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "history": "history"
    }
    }
    ```
    

    **1.4 微调模型**

    **配置微调参数**

    使用LoRA进行模型微调

    - **设置参数**


    - **设置输出并开始**

    完成执行后输出模型


    **合并模型**

    将 base model 与训练好的 LoRA Adapter 合并成一个新的模型

    - **合并模型配置**

    文件：`cust/merge_llama3_lora_sft.yaml`

    ```yaml
    ### Note: DO NOT use quantized model or quantization_bit when merging lora adapters
    
    ### model
    model_name_or_path: /root/autodl-tmp/models/Llama3-chinese
    adapter_name_or_path: /root/LLaMA-Factory/saves/LLaVA-NeXT-Llama3-8B-Chat/lora/train_2025-01-11-23-27-10
    template: llama3
    finetuning_type: lora
    
    ### export
    export_dir: /root/autodl-tmp/models/LLaMA3-8B-Chinese-Chat-merged
    export_size: 4
    export_device: cuda
    export_legacy_format:
    ```

    - **运行合并命令**

    ```bash
    llamafactory-cli export cust/merge_llama3_lora_sft.yaml
    ```

    ```bash
    root@autodl-container-e3b5468e80-8f05baad:~/autodl-tmp/models/LLaMA3-8B-Chinese-Chat-merged# ls -l
    total 15701044
    -rw-r--r-- 1 root root         750 Jan 11 23:47 config.json
    -rw-r--r-- 1 root root         147 Jan 11 23:47 generation_config.json
    -rw-r--r-- 1 root root 3986791344 Jan 11 23:47 model-00001-of-00005.safetensors
    -rw-r--r-- 1 root root 3926025416 Jan 11 23:47 model-00002-of-00005.safetensors
    -rw-r--r-- 1 root root 3926025440 Jan 11 23:47 model-00003-of-00005.safetensors
    -rw-r--r-- 1 root root 3171040864 Jan 11 23:47 model-00004-of-00005.safetensors
    -rw-r--r-- 1 root root 1050673280 Jan 11 23:47 model-00005-of-00005.safetensors
    -rw-r--r-- 1 root root       23993 Jan 11 23:47 model.safetensors.index.json
    -rw-r--r-- 1 root root         764 Jan 11 23:47 special_tokens_map.json
    -rw-r--r-- 1 root root    17208940 Jan 11 23:47 tokenizer.json
    -rw-r--r-- 1 root root      51581 Jan 11 23:47 tokenizer_config.json
    ```



    **1.5 微调推理**

    **Chat聊天**

    使用merged模型加载，并进行聊天

    **提问**

    在数据集中选择一个文件进行提问测试，实际效果还不错，和数据集中的内容相仿。




    **1.6 模型量化**

    - **量化配置**

    模型量化（Model Quantization）是一种将模型的参数和计算从高精度（通常是 32 位浮点数，FP32）转换为低精度（如 16 位浮点数，FP16，或者 8 位整数，INT8）的过程。


    - **执行量化**

    量化结果，生成的文件约6GB

    ```bash
    root@autodl-container-e3b5468e80-8f05baad:~/autodl-tmp/models/LLaMA3-8B-Chinese-Chat-q4# ls -l
    total 5619736
    -rw-r--r-- 1 root root   4301712 Jan 12 00:41 config.json
    -rw-r--r-- 1 root root       147 Jan 12 00:41 generation_config.json
    -rw-r--r-- 1 root root 4682270424 Jan 12 00:41 model-00001-of-00002.safetensors
    -rw-r--r-- 1 root root 1050673280 Jan 12 00:41 model-00002-of-00002.safetensors
    -rw-r--r-- 1 root root     78459 Jan 12 00:41 model.safetensors.index.json
    -rw-r--r-- 1 root root       512 Jan 12 00:41 special_tokens_map.json
    -rw-r--r-- 1 root root  17208940 Jan 12 00:41 tokenizer.json
    -rw-r--r-- 1 root root     51581 Jan 12 00:41 tokenizer_config.json
    ```

    - **推理验证**

3. **Unsloth**

    **2.1 环境依赖**

    ```bash
    conda create --name unsloth_env \
        python=3.11 \
        pytorch-cuda=12.1 \
        pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
        -y
    conda activate unsloth_env

    pip install unsloth
    ```

    **2.2 微调过程**

    **Qwen-7B-Instruct微调实战**

    详细代码请参考 `Unsloth_Finetuning_Qwen2.5_7B_Instruct_4bit.py`

      - **模型**：`4bit量化版的Qwen-7B-Instruct` 

      - **数据集**：`medical-o1-reasoning-SFT`，这是一个[医疗领域数据集](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT/viewer/zh?views%5B%5D=zh)，选择其中的中文数据作为训练集，共选择了1500条数据用于微调

      - **训练Loss结果**
    
        ![image-Unsloth-01](../Assets/Unsloth_Loss_2025-03-24_17-09-49.png)

    **2.3 评估指标和对比结果**

    详细代码请参考 `Evaluation.py`

      **BERTScore** 是一种基于 **BERT 语义相似度** 计算的文本评估方法，它用于衡量 **生成文本** 与 **参考文本** 之间的语义相似性。相比于传统的 **BLEU、ROUGE**，BERTScore **不依赖 n-gram 词匹配**，而是 **基于上下文的深度语义表示**，可以更准确地评估 NLP 任务（如机器翻译、摘要生成、文本生成等）的质量。


<br>

3. **LLaMA-Factory和Unsloth的区别**

- LLaMA-Factory支持WebUI可视化，便于操作
- Unsloth主打最快速、最低显存占用的微调工具
- LLaMA-Factory可以支持多GPU训练，Unsloth只能单GPU训练













# 模型试炼

## 模型
TinyLlama，约 2GB
## 数据集
MNIST Text Small，约 600B

使用加载的模型生成新的文本。生成的文本是基于输入文本的扩展，生成的最大 token 数量为 50
将生成的 token IDs 解码为可读文本，并打印原始样本和生成的结果


## 训练结果：
原始样本： 00 down ! ! ! ! ! ! ! ! ! ! ! ! ! !
01 down ! ! ! ! ! ! ! ! ! ! ! ! ! !
02 down ! ! ! ! ! ! % % C L a ^ ! !
03 down ! ! ! - ` ` ` ` ` Y ` Q ! !
04 down ! ! ! % ` ` ` R ^ ! ! ! ! !
05 down ! ! ! ! $ G ` ! ! ! ! ! ! !
06 down ! ! ! ! ! # ` Y < ! ! ! ! !
07 down ! ! ! ! ! ! 5 ` ` F ! ! ! !
08 down ! ! ! ! ! ! ! % ` ` 1 ! ! !
09 down ! ! ! ! ! ! F ` ` ` ! ! ! !
10 down ! ! ! ! 1 ` ` ` ` 4 ! ! ! !
11 down ! ! L ` ` ` ` 5 ! ! ! ! ! !
12 down ! ! ` ` V B ! ! ! ! ! ! ! !
13 down ! ! ! ! ! ! ! ! ! ! ! ! ! !
生成结果： 00 down ! ! ! ! ! ! ! ! ! ! ! ! ! !
01 down ! ! ! ! ! ! ! ! ! ! ! ! ! !
02 down ! ! ! ! ! ! % % C L a ^ ! !
03 down ! ! ! - ` ` ` ` ` Y ` Q ! !
04 down ! ! ! % ` ` ` R ^ ! ! ! ! !
05 down ! ! ! ! $ G ` ! ! ! ! ! ! !
06 down ! ! ! ! ! # ` Y < ! ! ! ! !
07 down ! ! ! ! ! ! 5 ` ` F ! ! ! !
08 down ! ! ! ! ! ! ! % ` ` 1 ! ! !
09 down ! ! ! ! ! ! F ` ` ` ! ! ! !
10 down ! ! ! ! 1 ` ` ` ` 4 ! ! ! !
11 down ! ! L ` ` ` ` 5 ! ! ! ! ! !
12 down ! ! ` ` V B ! ! ! ! ! ! ! !
13 down ! ! ! ! ! ! ! ! ! ! ! ! ! !
14 down ! ! ! ! ! ! ! ! ! ! ! ! ! !
15 down ! ! ! ! ! ! ! ! ! ! ! ! ! !
16 down ! ! ! ! ! ! ! ! ! !
