---
title: DVM快速入门：从Hello World到自定义算子
date: 2026-05-19 16:33:04
tags:
---

# DVM快速入门：从Hello World到自定义算子

## 前言

在深度学习领域，算子融合是提升模型性能的关键技术之一。传统的算子融合方案通常需要预先手写融合大算子或在编译阶段静态生成，这在动态Shape、动态图等场景下存在明显局限性。DVM（Device Virtual Machine）作为业界唯一的微秒级实时AI算子编译和执行框架，创新性地将算子编译过程从编译态延后到执行态，实现了根据运行时具体Shape进行实时算子编译和执行。

本文将带你快速入门DVM，从最简单的Hello World示例开始，逐步深入到自定义算子的开发。

---

## 一、DVM是什么？

### 1.1 核心定位

DVM是一个**微秒级实时AI算子编译和执行框架**，主要解决以下问题：

| 问题 | 传统方案 | DVM方案 |
|-----|---------|--------|
| 动态Shape支持 | 编译时无法确定Shape，只能使用shape-general策略 | 运行时根据具体Shape实时编译 |
| 编译性能 | 编译耗时秒级，与执行时间（微秒级）相差万倍 | 编译耗时降至百微秒级 |
| 融合覆盖范围 | 预先手写融合大算子，覆盖有限 | 自动融合，支持任意元算子组合 |

### 1.2 三层架构

DVM采用清晰的三层架构设计：

```
┌─────────────────────────────────────────────────────────┐
│                    用户层 (Host侧)                        │
│  基于元算子进行融合算子的计算逻辑构图表达                    │
│  API: Load, Store, Binary, Unary, Reduce, MatMul...     │
├─────────────────────────────────────────────────────────┤
│                   编译器层 (Host侧)                       │
│  实时编译生成以Tile为粒度的元算子字节码虚拟指令              │
│  流程: Shape推导 → Tiling策略 → 代码生成                  │
├─────────────────────────────────────────────────────────┤
│                   虚拟机层 (Device侧)                     │
│  解释执行DVM编译器生成的字节码虚拟指令                       │
│  核心: AIV核(Vector计算) + AIC核(Cube计算)               │
└─────────────────────────────────────────────────────────┘
```

### 1.3 核心优势

1. **微秒级编译**：大部分融合算子能在百微秒内完成编译
2. **原生动态Shape支持**：无需预先知道Shape信息
3. **高性能执行**：借助Ascend SIMD指令架构，性能可持平或超越手写算子
4. **灵活的融合能力**：支持Vector、Cube、Mix等多种融合模式

---

## 二、环境准备

### 2.1 系统要求

| 类别 | 要求 |
|-----|------|
| 硬件 | Host: X86/aarch64, Device: Ascend NPU (推荐A2/A3) |
| 操作系统 | Linux |
| CANN | 推荐8.3版本 |
| g++ | 推荐7.3.0 |
| Python | 推荐3.7+，需包含numpy、pybind11包 |

### 2.2 环境配置

**步骤1：配置CANN环境变量**

```bash
export ASCEND_CUSTOM_PATH=/<path_to_cann>
source $ASCEND_CUSTOM_PATH/ascend_toolkit/set_env.sh
```

**步骤2：配置DVM编译变量**

```bash
cd dvm
source env.sh
```

**步骤3：编译DVM**

```bash
# 编译生成libdvm.a和_dvm_py.so
make -j32

# 如果需要Debug版本
make dbg=1 -j32
```

编译成功后会生成：
- `libdvm.a`：DVM静态库（C++接口使用）
- `_dvm_py.so`：Python绑定库

---

## 三、Hello World：最简单的Add算子

让我们从最简单的Add算子开始，体验DVM的基本使用方式。

### 3.1 Python版本

```python
import numpy as np
import dvm

@dvm.kernel
def my_add(k, x, y):
    a = k.load(x, dvm.float32)
    b = k.load(y, dvm.float32)
    c = k.add(a, b)
    d = k.store(c)
    return d

x = np.full([32, 32], 0.1, np.float32)
y = np.full([32, 32], 0.3, np.float32)
z = my_add(x, y)
print(z)
```

### 3.2 代码解析

让我们逐行分析这段代码：

```python
@dvm.kernel
def my_add(k, x, y):
```

- `@dvm.kernel`：装饰器，将普通Python函数转换为DVM Kernel
- `k`：Kernel对象，由装饰器自动注入，用于构图
- `x, y`：占位符参数，调用时传入真实Tensor

```python
a = k.load(x, dvm.float32)
b = k.load(y, dvm.float32)
```

- `k.load()`：从全局内存加载数据到UB（Unified Buffer）
- `dvm.float32`：指定数据类型

```python
c = k.add(a, b)
```

- `k.add()`：执行加法操作

```python
d = k.store(c)
```

- `k.store()`：将结果存储回全局内存

### 3.3 执行流程

```
用户调用 my_add(x, y)
        ↓
装饰器自动处理：
  1. 创建Kernel对象
  2. 执行构图函数
  3. Normalize（Shape推导）
  4. CodeGen（字节码生成）
  5. Launch（下发执行）
        ↓
返回执行结果
```

### 3.4 动态Shape支持

DVM天然支持动态Shape，同一个Kernel可以处理不同Shape的输入：

```python
x = np.full([32, 32], 0.1, np.float32)
y = np.full([32, 32], 0.3, np.float32)
z = my_add(x, y)  # Shape: [32, 32]

x = np.full([32, 1], 0.2, np.float32)
y = np.full([32, 512], 0.7, np.float32)
z = my_add(x, y)  # Shape: [32, 512]，自动广播
```

---

## 四、进阶示例：BatchNorm算子

接下来我们实现一个更复杂的BatchNorm算子，展示DVM处理多算子融合的能力。

### 4.1 完整代码

```python
import numpy as np
import dvm

def np_bn(X, gamma, beta):
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    X_norm = (X - mean) / np.sqrt(var)
    out = gamma * X_norm + beta
    return out

@dvm.kernel
def bn_kernel(k, x, gamma, beta, rec_batch):
    x = k.load(x, dvm.float32)
    mean_sum = k.sum(x, (0,), False)
    rec_batch = k.scalar(rec_batch)
    mean = k.mul(mean_sum, rec_batch)

    x_sub = k.sub(x, mean)
    var_mul = k.mul(x_sub, x_sub)
    var_sum = k.sum(var_mul, (0,), False)
    var = k.mul(var_sum, rec_batch)

    norm_sqrt = k.sqrt(var)
    x_norm = k.div(x_sub, norm_sqrt)

    gamma = k.load(gamma, dvm.float32)
    beta = k.load(beta, dvm.float32)
    out_mul = k.mul(gamma, x_norm)
    out_add = k.add(out_mul, beta)
    out = k.store(out_add)
    return out

def my_bn(x, gamma, beta, batch_size):
    return bn_kernel(x, gamma, beta, 1.0 / batch_size)

input_x = np.random.normal(0, 1, (32, 1000)).astype(np.float32)
gamma = np.random.normal(0, 1, [1000]).astype(np.float32)
beta = np.random.normal(0, 1, [1000]).astype(np.float32)
out = my_bn(input_x, gamma, beta, 32)

expect = np_bn(input_x, gamma, beta)
assert np.allclose(out, expect, rtol=1e-3, atol=1e-3)
```

### 4.2 关键API解析

**Reduce操作**

```python
mean_sum = k.sum(x, (0,), False)
```

- 第一个参数：输入张量
- 第二个参数：reduce的维度，元组形式
- 第三个参数：是否保持维度

**标量操作**

```python
rec_batch = k.scalar(rec_batch)
mean = k.mul(mean_sum, rec_batch)
```

- `k.scalar()`：将Python标量转换为DVM标量
- 可以与张量进行运算

### 4.3 融合效果

BatchNorm包含多个操作：
1. 计算均值：sum → mul
2. 计算方差：sub → mul → sum → mul
3. 归一化：sub → div
4. 缩放平移：mul → add

DVM会将这些操作自动融合为一个Kernel，避免中间结果的反复读写，大幅提升性能。

---

## 五、高级示例：Attention算子

Attention是Transformer的核心组件，让我们看看如何用DVM实现。

### 5.1 完整代码

```python
import math
import numpy as np
import dvm

def dvm_softmax(k, x, axis):
    x = k.cast(x, dvm.float32)
    max_x = k.max(x, (axis,), True)
    x = k.sub(x, max_x)
    exp_x = k.exp(x)
    denom = k.sum(exp_x, (axis,), True)
    out = k.div(exp_x, denom)
    return k.cast(out, dvm.float16)

@dvm.kernel
def attention_kernel(k, q, k_mat, v, scale):
    q = k.load(q, dvm.float16)
    k_mat = k.load(k_mat, dvm.float16)
    v = k.load(v, dvm.float16)
    scale = k.scalar(scale)
    
    scores = k.matmul(q, k_mat, False, True)
    scores = k.mul(scores, scale)
    probs = dvm_softmax(k, scores, 1)
    out = k.matmul(probs, v, False, False)
    out = k.store(out)
    return out

q = np.random.normal(0, 0.02, (4, 8)).astype(np.float16)
k_mat = np.random.normal(0, 0.02, (6, 8)).astype(np.float16)
v = np.random.normal(0, 0.02, (6, 12)).astype(np.float16)
scale = 1.0 / math.sqrt(q.shape[-1])

out = attention_kernel(q, k_mat, v, scale)
```

### 5.2 MatMul操作

```python
scores = k.matmul(q, k_mat, False, True)
```

- 第一个参数：左矩阵
- 第二个参数：右矩阵
- 第三个参数：是否转置左矩阵
- 第四个参数：是否转置右矩阵

### 5.3 类型转换

```python
x = k.cast(x, dvm.float32)
```

- 支持多种数据类型转换：float16 ↔ float32, bfloat16 ↔ float32等

### 5.4 子函数封装

DVM支持将常用操作封装为Python函数：

```python
def dvm_softmax(k, x, axis):
    x = k.cast(x, dvm.float32)
    max_x = k.max(x, (axis,), True)
    x = k.sub(x, max_x)
    exp_x = k.exp(x)
    denom = k.sum(exp_x, (axis,), True)
    out = k.div(exp_x, denom)
    return k.cast(out, dvm.float16)
```

这样可以在多个Kernel中复用。

---

## 六、MatMul后融合示例

DVM支持MatMul（Cube计算）与后续Vector计算的融合，这是提升性能的关键优化。

### 6.1 完整代码

```python
import numpy as np
import dvm

def dvm_silu(k, x):
    neg = k.mul(x, -1.0)
    exp_neg = k.exp(neg)
    denom = k.add(exp_neg, 1.0)
    sigmoid = k.div(1.0, denom)
    return k.mul(x, sigmoid)

@dvm.kernel
def matmul_post_fusion(k, a, b, bias, scale, shift):
    a = k.load(a, dvm.float16)
    b = k.load(b, dvm.float16)
    bias = k.load(bias, dvm.float32)
    scale = k.scalar(scale)
    shift = k.scalar(shift)
    
    out = k.matmul(a, b, False, False)
    out = k.cast(out, dvm.float32)
    out = k.add(out, bias)
    out = k.mul(out, scale)
    out = k.add(out, shift)
    out = dvm_silu(k, out)
    out = k.store(out)
    return out

m, k_dim, n = 32, 64, 48
a = np.random.normal(0, 0.02, (m, k_dim)).astype(np.float16)
b = np.random.normal(0, 0.02, (k_dim, n)).astype(np.float16)
bias = np.random.normal(0, 0.02, (n,)).astype(np.float32)
scale = 0.5
shift = -0.1

out = matmul_post_fusion(a, b, bias, scale, shift)
```

### 6.2 Mix Kernel原理

```
传统方式：
MatMul → 写回GM → 读入UB → Bias → Scale → Shift → SiLU → 写回GM
（多次GM访问，性能差）

DVM Mix融合：
MatMul → [L1 Cache] → Bias → Scale → Shift → SiLU → 写回GM
（Cube和Vector在Tile粒度Overlap，减少GM访问）
```

---

## 七、C++接口使用

对于需要更高性能或更细粒度控制的场景，可以使用C++接口。

### 7.1 完整示例

```cpp
#include <iostream>
#include <vector>
#include "acl/acl_rt.h"
#include "dvm.h"

int main() {
    aclrtSetDevice(0);
    
    float *host_x = new float[1024];
    float *host_y = new float[1024];
    for (size_t i = 0; i < 1024; ++i) {
        host_x[i] = 2.0f;
        host_y[i] = 3.0f;
    }
    
    void *dev_x, *dev_y, *dev_out;
    aclrtMalloc(&dev_x, 1024 * sizeof(float), ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    aclrtMalloc(&dev_y, 1024 * sizeof(float), ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    aclrtMalloc(&dev_out, 1024 * sizeof(float), ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    
    aclrtMemcpyAsync(dev_x, 1024 * sizeof(float), host_x, 
                     1024 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, nullptr);
    aclrtMemcpyAsync(dev_y, 1024 * sizeof(float), host_y, 
                     1024 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, nullptr);
    
    // 定义DVM Kernel
    dvm::Kernel kernel;
    std::vector<int64_t> shape_data = {1024};
    dvm::IntArrayRef shape_ref(shape_data);
    
    kernel.Reset(dvm::kVector, 0);
    auto x = kernel.Load(dev_x, &shape_ref, dvm::kFloat32);
    auto y = kernel.Load(dev_y, &shape_ref, dvm::kFloat32);
    auto z = kernel.Binary<dvm::kMul>(x, y);
    auto r = kernel.Binary<dvm::kAdd>(z, 0.5f);
    auto out = kernel.Store(dev_out, r);
    
    // 编译执行
    kernel.CodeGen();
    kernel.Launch(nullptr, 0, 0, nullptr);
    
    aclrtSynchronizeStream(nullptr);
    
    // 清理资源
    aclrtFree(dev_x);
    aclrtFree(dev_y);
    aclrtFree(dev_out);
    delete[] host_x;
    delete[] host_y;
    
    return 0;
}
```

### 7.2 关键步骤

```cpp
// 1. 创建并重置Kernel
dvm::Kernel kernel;
kernel.Reset(dvm::kVector, 0);  // kVector类型，静态Shape

// 2. 构图
auto x = kernel.Load(dev_x, &shape_ref, dvm::kFloat32);
auto y = kernel.Load(dev_y, &shape_ref, dvm::kFloat32);
auto z = kernel.Binary<dvm::kMul>(x, y);
auto out = kernel.Store(dev_out, z);

// 3. 编译
kernel.CodeGen();

// 4. 执行
kernel.Launch(nullptr, 0, 0, stream);
```

### 7.3 动态Shape处理

```cpp
dvm::Kernel kernel;
kernel.Reset(dvm::kVector, dvm::kDynamic);  // 标记为动态Shape

// 构图时使用IntArrayRef指针
dvm::IntArrayRef shape_ref;
auto x = kernel.Load(nullptr, &shape_ref, dvm::kFloat32);  // 地址先传nullptr

// 执行时更新Shape和地址
std::vector<int64_t> shape = {32, 1024};
shape_ref = shape;

kernel.Normalize();  // Shape推导

// 地址重定位
std::vector<dvm::RelocEntry> relocs;
relocs.emplace_back(x, tensor_a.addr());
relocs.emplace_back(out, tensor_c.addr());

kernel.CodeGen(relocs.data(), relocs.size(), ws_alloc);
kernel.Launch(stream);
```

---

## 八、调试技巧

### 8.1 Dump元算子构图

```python
@dvm.kernel
def my_kernel(k, x):
    # ... 构图代码 ...
    pass

# 在调用前设置调试模式
my_kernel.debug = True
result = my_kernel(x)
```

### 8.2 C++版本调试

```cpp
// 打印元算子构图
std::cout << kernel.Dump() << std::endl;

// 打印字节码反汇编
kernel.CodeGen();
std::cout << kernel.Das() << std::endl;
```

### 8.3 使用Debug版本

```bash
make dbg=1 -j32
```

Debug版本会启用更多断言检查，帮助定位问题。

---

## 九、API速查表

### 9.1 数据类型

| DVM类型 | 说明 |
|--------|------|
| dvm.bool | 布尔类型 |
| dvm.float16 | 半精度浮点 |
| dvm.bfloat16 | Brain Float16 |
| dvm.float32 | 单精度浮点 |
| dvm.int32 | 32位整数 |

### 9.2 构图API

| API | 说明 |
|-----|------|
| `k.load(tensor, dtype)` | 加载张量 |
| `k.store(tensor)` | 存储结果 |
| `k.add(a, b)` | 加法 |
| `k.sub(a, b)` | 减法 |
| `k.mul(a, b)` | 乘法 |
| `k.div(a, b)` | 除法 |
| `k.sqrt(x)` | 平方根 |
| `k.exp(x)` | 指数 |
| `k.sum(x, axis, keepdims)` | 求和 |
| `k.max(x, axis, keepdims)` | 最大值 |
| `k.min(x, axis, keepdims)` | 最小值 |
| `k.matmul(a, b, trans_a, trans_b)` | 矩阵乘法 |
| `k.cast(x, dtype)` | 类型转换 |

### 9.3 Kernel类型

| 类型 | 说明 | 使用场景 |
|-----|------|---------|
| kVector | 纯Vector计算 | 元素级操作、Reduce |
| kCube | 纯MatMul计算 | 矩阵乘法 |
| kMix | Cube+Vector融合 | MatMul后接激活函数 |

---

## 十、总结

本文通过四个渐进式示例，带你快速入门了DVM的核心使用方式：

1. **Hello World**：掌握了基本的Load/Add/Store操作
2. **BatchNorm**：学会了多算子融合和Reduce操作
3. **Attention**：理解了MatMul和Softmax的组合使用
4. **MatMul后融合**：体验了Cube和Vector的混合融合

### 关键要点

1. **装饰器模式**：`@dvm.kernel`自动处理编译执行流程
2. **元算子构图**：通过Load/Store/计算API构建计算图
3. **动态Shape**：天然支持，无需特殊处理
4. **自动融合**：多个操作自动融合为单一Kernel

### 下一步学习

- 深入理解Kernel对象的内部实现
- 学习Tiling策略和字节码生成
- 探索虚拟机的执行机制
- 研究图优化Pass的实现

DVM为AI算子优化提供了一个强大而灵活的平台，希望本文能帮助你快速上手，在实际项目中发挥其价值。

---

## 参考资料

- [DVM官方仓库](https://gitcode.com/mindspore/dvm)
- [用户开发指南](../docs/tutorial.md)
- [示例代码](../examples/)
