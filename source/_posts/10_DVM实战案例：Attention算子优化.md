---
title: DVM实战案例：Attention算子优化
date: 2026-05-19 16:33:04
tags:
---

# DVM实战案例：Attention算子优化

## 前言

Attention是Transformer架构的核心组件，其计算复杂度和访存开销对模型性能影响巨大。本文将以Attention算子为例，展示如何使用DVM进行算子优化，涵盖从基础实现到高级优化的完整过程。

---

## 一、Attention算子分析

### 1.1 标准Attention公式

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

其中：
- Q: [batch, seq_len, d_k] 查询矩阵
- K: [batch, seq_len, d_k] 键矩阵
- V: [batch, seq_len, d_v] 值矩阵
- d_k: 查询/键的维度
```

### 1.2 计算分解

```
步骤1: scores = Q @ K^T          # [batch, seq_len, seq_len]
步骤2: scores = scores / scale   # [batch, seq_len, seq_len]
步骤3: probs = softmax(scores)   # [batch, seq_len, seq_len]
步骤4: output = probs @ V        # [batch, seq_len, d_v]
```

### 1.3 性能瓶颈

| 瓶颈 | 说明 |
|-----|------|
| MatMul密集 | 两次矩阵乘法，计算量大 |
| Softmax访存 | 需要多次读写scores矩阵 |
| 中间结果大 | scores矩阵为 seq_len × seq_len |

---

## 二、基础实现

### 2.1 Python实现

```python
import math
import numpy as np
import dvm

def dvm_softmax(k, x, axis):
    """Softmax实现"""
    x = k.cast(x, dvm.float32)
    max_x = k.max(x, (axis,), True)
    x = k.sub(x, max_x)
    exp_x = k.exp(x)
    denom = k.sum(exp_x, (axis,), True)
    out = k.div(exp_x, denom)
    return k.cast(out, dvm.float16)

@dvm.kernel
def attention_basic(k, q, k_mat, v, scale):
    """基础Attention实现"""
    q = k.load(q, dvm.float16)
    k_mat = k.load(k_mat, dvm.float16)
    v = k.load(v, dvm.float16)
    scale = k.scalar(scale)
    
    # Q @ K^T
    scores = k.matmul(q, k_mat, False, True)
    
    # Scale
    scores = k.mul(scores, scale)
    
    # Softmax
    probs = dvm_softmax(k, scores, 1)
    
    # Probs @ V
    output = k.matmul(probs, v, False, False)
    
    return k.store(output)

# 测试
batch, seq_len, d_k, d_v = 1, 1024, 64, 64
q = np.random.randn(batch, seq_len, d_k).astype(np.float16)
k = np.random.randn(batch, seq_len, d_k).astype(np.float16)
v = np.random.randn(batch, seq_len, d_v).astype(np.float16)
scale = 1.0 / math.sqrt(d_k)

output = attention_basic(q, k, v, scale)
```

### 2.2 C++实现

```cpp
#include "dvm.h"

void AttentionBasic(dvm::Kernel &kernel, void *q, void *k, void *v, void *out,
                    int64_t batch, int64_t seq_len, int64_t d_k, int64_t d_v,
                    float scale) {
  std::vector<int64_t> qk_shape = {batch, seq_len, d_k};
  std::vector<int64_t> v_shape = {batch, seq_len, d_v};
  std::vector<int64_t> scores_shape = {batch, seq_len, seq_len};
  
  dvm::IntArrayRef qk_ref(qk_shape);
  dvm::IntArrayRef v_ref(v_shape);
  dvm::IntArrayRef scores_ref(scores_shape);
  
  kernel.Reset(dvm::kSplit, dvm::kDynamic);
  
  // Load
  auto q_obj = kernel.Load(q, &qk_ref, dvm::kFloat16);
  auto k_obj = kernel.Load(k, &qk_ref, dvm::kFloat16);
  auto v_obj = kernel.Load(v, &v_ref, dvm::kFloat16);
  
  // Q @ K^T
  auto scores = kernel.MatMul(q_obj, k_obj, false, true, nullptr);
  
  // Scale
  auto scale_obj = kernel.Scalar(scale);
  auto scaled_scores = kernel.Binary<dvm::kMul>(scores, scale_obj);
  
  // Softmax
  auto max_scores = kernel.Flex<dvm::kMax>(scaled_scores, &scores_ref, true);
  auto shifted = kernel.Binary<dvm::kSub>(scaled_scores, max_scores);
  auto exp_scores = kernel.Unary<dvm::kExp>(shifted);
  auto sum_exp = kernel.Flex<dvm::kSum>(exp_scores, &scores_ref, true);
  auto probs = kernel.Binary<dvm::kDiv>(exp_scores, sum_exp);
  
  // Probs @ V
  auto output = kernel.MatMul(probs, v_obj, false, false, nullptr);
  
  // Store
  kernel.Store(out, output);
  
  kernel.Normalize();
}
```

---

## 三、Flash Attention优化

### 3.1 Flash Attention原理

Flash Attention通过Tiling和重计算减少HBM访问：

```
传统Attention：
Q, K, V -> HBM -> scores -> HBM -> probs -> HBM -> output
         (O(N^2) HBM访问)

Flash Attention：
Q, K, V -> 分块处理 -> output
         (O(N) HBM访问)
```

### 3.2 Tiling策略

```python
@dvm.kernel(ktype="split")
def flash_attention(k, q, k_mat, v, scale, block_size=64):
    """Flash Attention实现"""
    q = k.load(q, dvm.float16)
    k_mat = k.load(k_mat, dvm.float16)
    v = k.load(v, dvm.float16)
    scale = k.scalar(scale)
    
    seq_len = q.shape()[0]
    
    # 分块处理
    for i in range(0, seq_len, block_size):
        # 当前Q块
        q_block = k.slice(q, i, i + block_size)
        
        # 初始化累加器
        o_acc = k.zeros((block_size, d_v), dvm.float32)
        l_acc = k.zeros((block_size, 1), dvm.float32)
        m_acc = k.full((block_size, 1), -float('inf'), dvm.float32)
        
        for j in range(0, seq_len, block_size):
            # 当前K, V块
            k_block = k.slice(k_mat, j, j + block_size)
            v_block = k.slice(v, j, j + block_size)
            
            # 计算scores
            scores = k.matmul(q_block, k_block, False, True)
            scores = k.mul(scores, scale)
            
            # 更新最大值
            m_new = k.maximum(m_acc, k.max(scores, -1, True))
            
            # 重归一化
            p = k.exp(k.sub(scores, m_new))
            l_new = k.add(
                k.mul(l_acc, k.exp(k.sub(m_acc, m_new))),
                k.sum(p, -1, True)
            )
            
            # 更新输出
            o_acc = k.mul(o_acc, k.exp(k.sub(m_acc, m_new)))
            o_acc = k.add(o_acc, k.matmul(p, v_block))
            
            # 更新累加器
            m_acc = m_new
            l_acc = l_new
        
        # 归一化输出
        o_block = k.div(o_acc, l_acc)
        k.store_slice(o_block, i, i + block_size)
```

### 3.3 DVM Mix Kernel优化

```python
@dvm.kernel(ktype="mix")
def attention_mix(k, q, k_mat, v, scale):
    """使用Mix Kernel优化MatMul后融合"""
    q = k.load(q, dvm.float16)
    k_mat = k.load(k_mat, dvm.float16)
    v = k.load(v, dvm.float16)
    scale = k.scalar(scale)
    
    # 第一个MatMul: Q @ K^T
    # 后续Vector操作: Scale, Softmax
    scores = k.matmul(q, k_mat, False, True)
    scores = k.mul(scores, scale)
    
    # Softmax (Vector操作自动融合到MatMul后)
    scores = k.cast(scores, dvm.float32)
    max_scores = k.max(scores, (1,), True)
    scores = k.sub(scores, max_scores)
    exp_scores = k.exp(scores)
    sum_exp = k.sum(exp_scores, (1,), True)
    probs = k.div(exp_scores, sum_exp)
    probs = k.cast(probs, dvm.float16)
    
    # 第二个MatMul: Probs @ V
    output = k.matmul(probs, v, False, False)
    
    return k.store(output)
```

---

## 四、Multi-Head Attention实现

### 4.1 MHA结构

```
MultiHeadAttention(Q, K, V):
    # 分头
    Q_heads = split(Q, num_heads)  # [batch, num_heads, seq_len, d_k]
    K_heads = split(K, num_heads)
    V_heads = split(V, num_heads)
    
    # 每个头独立计算
    for h in range(num_heads):
        head_output[h] = Attention(Q_heads[h], K_heads[h], V_heads[h])
    
    # 合并
    output = concat(head_output)  # [batch, seq_len, d_model]
    output = output @ W_o
```

### 4.2 DVM实现

```python
@dvm.kernel
def multi_head_attention(k, q, k_mat, v, w_o, scale, num_heads):
    """Multi-Head Attention实现"""
    batch, seq_len, d_model = q.shape()
    d_k = d_model // num_heads
    
    q = k.load(q, dvm.float16)
    k_mat = k.load(k_mat, dvm.float16)
    v = k.load(v, dvm.float16)
    w_o = k.load(w_o, dvm.float16)
    scale = k.scalar(scale)
    
    # Reshape为多头
    q = k.reshape(q, (batch, num_heads, seq_len, d_k))
    k_mat = k.reshape(k_mat, (batch, num_heads, seq_len, d_k))
    v = k.reshape(v, (batch, num_heads, seq_len, d_k))
    
    # Q @ K^T / scale
    scores = k.matmul(q, k_mat, False, True)
    scores = k.mul(scores, scale)
    
    # Softmax
    scores = k.cast(scores, dvm.float32)
    max_scores = k.max(scores, (3,), True)
    scores = k.sub(scores, max_scores)
    exp_scores = k.exp(scores)
    sum_exp = k.sum(exp_scores, (3,), True)
    probs = k.div(exp_scores, sum_exp)
    probs = k.cast(probs, dvm.float16)
    
    # Probs @ V
    output = k.matmul(probs, v, False, False)
    
    # Reshape回原始形状
    output = k.reshape(output, (batch, seq_len, d_model))
    
    # Output projection
    output = k.matmul(output, w_o, False, False)
    
    return k.store(output)
```

---

## 五、性能优化技巧

### 5.1 数据类型优化

```python
# 使用BF16提高数值稳定性
@dvm.kernel
def attention_bf16(k, q, k_mat, v, scale):
    q = k.load(q, dvm.bfloat16)
    k_mat = k.load(k_mat, dvm.bfloat16)
    v = k.load(v, dvm.bfloat16)
    
    # MatMul使用BF16
    scores = k.matmul(q, k_mat, False, True)
    
    # Softmax使用FP32保证精度
    scores = k.cast(scores, dvm.float32)
    max_scores = k.max(scores, (1,), True)
    scores = k.sub(scores, max_scores)
    exp_scores = k.exp(scores)
    sum_exp = k.sum(exp_scores, (1,), True)
    probs = k.div(exp_scores, sum_exp)
    probs = k.cast(probs, dvm.bfloat16)
    
    output = k.matmul(probs, v, False, False)
    return k.store(output)
```

### 5.2 自定义Tiling

```python
@dvm.kernel
def attention_custom_tile(k, q, k_mat, v, scale):
    q = k.load(q, dvm.float16)
    k_mat = k.load(k_mat, dvm.float16)
    v = k.load(v, dvm.float16)
    scale = k.scalar(scale)
    
    # 设置自定义Tiling
    k.set_tile(start=0, end=2, num=128, factor=0)
    
    scores = k.matmul(q, k_mat, False, True)
    scores = k.mul(scores, scale)
    
    # Softmax
    scores = k.cast(scores, dvm.float32)
    max_scores = k.max(scores, (1,), True)
    scores = k.sub(scores, max_scores)
    exp_scores = k.exp(scores)
    sum_exp = k.sum(exp_scores, (1,), True)
    probs = k.div(exp_scores, sum_exp)
    probs = k.cast(probs, dvm.float16)
    
    output = k.matmul(probs, v, False, False)
    return k.store(output)
```

### 5.3 图优化Pass应用

```python
from dvm.tester import Tester

def test_attention_with_pass():
    t = Tester("split")
    
    q = np.random.randn(1, 1024, 64).astype(np.float16)
    k = np.random.randn(1, 1024, 64).astype(np.float16)
    v = np.random.randn(1, 1024, 64).astype(np.float16)
    
    q_obj = t.load(q)
    k_obj = t.load(k)
    v_obj = t.load(v)
    
    # 构图
    scores = t.matmul(q_obj, k_obj, False, True)
    scores = t.mul(scores, 1.0 / 8.0)
    
    # Softmax
    scores = t.cast(scores, dvm.float32)
    max_scores = t.max(scores, (2,), True)
    scores = t.sub(scores, max_scores)
    exp_scores = t.exp(scores)
    sum_exp = t.sum(exp_scores, (2,), True)
    probs = t.div(exp_scores, sum_exp)
    probs = t.cast(probs, dvm.float16)
    
    output = t.matmul(probs, v_obj, False, False)
    t.store(output)
    
    # 设置优化Pass
    t.set_passes("ReorderStore", "CompactPeakLiveness", "EliminateReshape")
    
    # 编译执行
    t.codegen()
    t.run()
```

---

## 六、性能测试与分析

### 6.1 测试代码

```python
import time
import numpy as np
import dvm
from dvm.tester import Tester

def benchmark_attention(seq_len, d_k, d_v, num_runs=100):
    """性能测试"""
    q = np.random.randn(1, seq_len, d_k).astype(np.float16)
    k = np.random.randn(1, seq_len, d_k).astype(np.float16)
    v = np.random.randn(1, seq_len, d_v).astype(np.float16)
    scale = 1.0 / np.sqrt(d_k)
    
    # 预热
    for _ in range(10):
        output = attention_basic(q, k, v, scale)
    
    # 测试
    times = []
    for _ in range(num_runs):
        start = time.time()
        output = attention_basic(q, k, v, scale)
        end = time.time()
        times.append((end - start) * 1000)  # ms
    
    return {
        'min': min(times),
        'max': max(times),
        'mean': np.mean(times),
        'std': np.std(times),
    }

# 运行测试
results = {}
for seq_len in [256, 512, 1024, 2048]:
    results[seq_len] = benchmark_attention(seq_len, 64, 64)
    print(f"seq_len={seq_len}: {results[seq_len]['mean']:.3f} ms")
```

### 6.2 性能对比

```python
def compare_implementations():
    """对比不同实现的性能"""
    seq_len = 1024
    d_k = d_v = 64
    
    q = np.random.randn(1, seq_len, d_k).astype(np.float16)
    k = np.random.randn(1, seq_len, d_k).astype(np.float16)
    v = np.random.randn(1, seq_len, d_v).astype(np.float16)
    scale = 1.0 / np.sqrt(d_k)
    
    # PyTorch实现
    import torch
    import torch.nn.functional as F
    
    q_torch = torch.from_numpy(q).cuda()
    k_torch = torch.from_numpy(k).cuda()
    v_torch = torch.from_numpy(v).cuda()
    
    # 预热
    for _ in range(10):
        F.scaled_dot_product_attention(q_torch, k_torch, v_torch)
    
    # 测试PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        F.scaled_dot_product_attention(q_torch, k_torch, v_torch)
    torch.cuda.synchronize()
    end = time.time()
    torch_time = (end - start) * 1000 / 100
    
    # 测试DVM
    dvm_time = benchmark_attention(seq_len, d_k, d_v)['mean']
    
    print(f"PyTorch: {torch_time:.3f} ms")
    print(f"DVM: {dvm_time:.3f} ms")
    print(f"Speedup: {torch_time / dvm_time:.2f}x")
```

### 6.3 Profiling分析

```python
def profile_attention():
    """Profiling分析"""
    t = Tester("split")
    
    q = np.random.randn(1, 1024, 64).astype(np.float16)
    k = np.random.randn(1, 1024, 64).astype(np.float16)
    v = np.random.randn(1, 1024, 64).astype(np.float16)
    
    # ... 构图 ...
    
    # 运行Profiling
    t.run_msprof("./profiling_output", test_num=100)
    
    # 分析结果
    # 使用msprof工具分析生成的数据
```

---

## 七、完整示例

### 7.1 带Mask的Attention

```python
@dvm.kernel
def attention_with_mask(k, q, k_mat, v, mask, scale):
    """带Mask的Attention"""
    q = k.load(q, dvm.float16)
    k_mat = k.load(k_mat, dvm.float16)
    v = k.load(v, dvm.float16)
    mask = k.load(mask, dvm.bool_)
    scale = k.scalar(scale)
    
    # Q @ K^T
    scores = k.matmul(q, k_mat, False, True)
    scores = k.mul(scores, scale)
    
    # Apply mask
    scores = k.cast(scores, dvm.float32)
    scores = k.select(mask, scores, -1e9)
    
    # Softmax
    max_scores = k.max(scores, (1,), True)
    scores = k.sub(scores, max_scores)
    exp_scores = k.exp(scores)
    sum_exp = k.sum(exp_scores, (1,), True)
    probs = k.div(exp_scores, sum_exp)
    probs = k.cast(probs, dvm.float16)
    
    # Probs @ V
    output = k.matmul(probs, v, False, False)
    
    return k.store(output)
```

### 7.2 Cross Attention

```python
@dvm.kernel
def cross_attention(k, q, k_mat, v, scale):
    """Cross Attention (Encoder-Decoder Attention)"""
    # Q来自Decoder: [batch, tgt_len, d_k]
    # K, V来自Encoder: [batch, src_len, d_k/d_v]
    
    q = k.load(q, dvm.float16)      # [batch, tgt_len, d_k]
    k_mat = k.load(k_mat, dvm.float16)  # [batch, src_len, d_k]
    v = k.load(v, dvm.float16)      # [batch, src_len, d_v]
    scale = k.scalar(scale)
    
    # Q @ K^T: [batch, tgt_len, src_len]
    scores = k.matmul(q, k_mat, False, True)
    scores = k.mul(scores, scale)
    
    # Softmax
    scores = k.cast(scores, dvm.float32)
    max_scores = k.max(scores, (2,), True)
    scores = k.sub(scores, max_scores)
    exp_scores = k.exp(scores)
    sum_exp = k.sum(exp_scores, (2,), True)
    probs = k.div(exp_scores, sum_exp)
    probs = k.cast(probs, dvm.float16)
    
    # Probs @ V: [batch, tgt_len, d_v]
    output = k.matmul(probs, v, False, False)
    
    return k.store(output)
```

### 7.3 Grouped Query Attention (GQA)

```python
@dvm.kernel
def grouped_query_attention(k, q, k_mat, v, scale, num_groups):
    """Grouped Query Attention"""
    batch, seq_len, d_model = q.shape()
    num_heads = d_model * 2 // d_model  # 假设d_k = d_v
    heads_per_group = num_heads // num_groups
    
    q = k.load(q, dvm.float16)
    k_mat = k.load(k_mat, dvm.float16)
    v = k.load(v, dvm.float16)
    scale = k.scalar(scale)
    
    # Reshape
    q = k.reshape(q, (batch, num_heads, seq_len, d_model // num_heads))
    k_mat = k.reshape(k_mat, (batch, num_groups, seq_len, d_model // num_heads))
    v = k.reshape(v, (batch, num_groups, seq_len, d_model // num_heads))
    
    # 扩展K, V以匹配Q的头数
    # 这里需要广播操作
    
    # 计算Attention
    # ...
    
    return k.store(output)
```

---

## 八、最佳实践

### 8.1 实现建议

| 建议 | 说明 |
|-----|------|
| 使用Mix Kernel | MatMul后接Vector操作自动融合 |
| 选择合适的数据类型 | BF16用于训练，FP16用于推理 |
| 自定义Tiling | 根据问题规模调整Tile大小 |
| 启用图优化 | 使用ReorderStore等Pass优化 |

### 8.2 调试技巧

```python
# Dump计算图
@dvm.kernel
def debug_attention(k, q, k_mat, v, scale):
    q = k.load(q, dvm.float16)
    k_mat = k.load(k_mat, dvm.float16)
    v = k.load(v, dvm.float16)
    
    scores = k.matmul(q, k_mat, False, True)
    
    # 打印中间结果Shape
    print(f"scores shape: {scores.shape()}")
    
    scores = k.mul(scores, scale)
    
    # ... 后续操作 ...
    
    return k.store(output)

# 反汇编字节码
output = debug_attention(q, k, v, scale)
print(debug_attention.das())
```

### 8.3 常见问题

```python
# 问题1: 数值溢出
# 解决: 使用FP32进行Softmax计算
scores = k.cast(scores, dvm.float32)  # 转换为FP32
# ... softmax计算 ...
probs = k.cast(probs, dvm.float16)    # 转回FP16

# 问题2: 内存不足
# 解决: 使用Split Kernel自动拆分
@dvm.kernel(ktype="split")
def attention_split(...):
    pass

# 问题3: 性能不佳
# 解决: 使用Tuning优化
from dvm.tuner import Tuner
tuner = Tuner()
tuner.tune(attention_kernel)
```

---

## 九、总结

本文通过Attention算子的优化实践，展示了DVM的强大能力：

1. **基础实现**：理解Attention的计算流程
2. **Flash Attention**：Tiling和重计算优化
3. **Mix Kernel**：MatMul与Vector操作融合
4. **Multi-Head Attention**：多头注意力实现
5. **性能优化**：数据类型、Tiling、图优化
6. **性能测试**：Benchmark和Profiling
7. **高级变体**：Masked Attention、Cross Attention、GQA

DVM为Attention算子优化提供了灵活高效的工具，通过合理的策略可以实现显著的性能提升。

---

## 参考资料

- [DVM官方仓库](https://gitcode.com/mindspore/dvm)
- [Flash Attention论文](https://arxiv.org/abs/2205.14135)
- [examples/03_attention.py](../examples/03_attention.py)
- [用户开发指南](../docs/tutorial.md)
