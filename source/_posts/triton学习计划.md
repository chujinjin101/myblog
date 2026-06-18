---
title: triton学习计划
date: 2026-04-21 16:33:04
tags:
---


# 🧭 总体阶段划分

| 阶段 | 周数   | 目标               |
| -- | ---- | ---------------- |
| 基础 | 1-2  | 会写 Triton kernel |
| 进阶 | 3-5  | 掌握并行与内存优化        |
| 编译 | 6-8  | 吃透 JIT & IR      |
| 专家 | 9-10 | 能做性能工程           |

---

# 📅 第 1 周：环境 + 基础认知

## 🎯 目标

* 跑通 Triton
* 理解 kernel 执行模型

## 📚 学习内容

* 安装：`pip install triton`
* 官方 tutorial（vector add）

重点理解：

* `@triton.jit`
* `program_id`
* `grid`

---

## 🧪 实践

实现：

* 向量加法
* 向量乘法

---

## 📖 推荐资料

* Triton 官方文档（Intro）
* Triton GitHub examples

---

# 📅 第 2 周：内存与并行模型

## 🎯 目标

* 理解 Triton 的并行方式
* 掌握内存访问

---

## 📚 学习重点

* `tl.load / tl.store`
* mask（避免越界）
* BLOCK_SIZE 设计

核心概念：

👉 memory coalescing（连续访问）

---

## 🧪 实践

实现：

* 向量 reduce（sum）
* 带 mask 的 kernel

---

## 📖 推荐资料

* CUDA memory coalescing 文章（通用）

---

# 📅 第 3 周：性能基础（GPU视角）

## 🎯 目标

* 理解 GPU 为什么快/慢

---

## 📚 学习重点

* warp / SM / occupancy
* latency hiding
* memory hierarchy

建议了解 NVIDIA GPU（哪怕你不写 CUDA）

---

## 🧪 实践

* 改 BLOCK_SIZE 做 benchmark
* 分析性能变化

---

## 📖 推荐资料

* NVIDIA CUDA Programming Guide（重点章节）
* GPU 架构文章

---

# 📅 第 4 周：进阶 kernel（算子级）

## 🎯 目标

* 能写常见 DL 算子

---

## 🧪 实践（很关键）

实现：

* softmax
* layernorm
* matmul（简化版）

---

## 📚 学习重点

* reduction pattern
* 数值稳定性（softmax）

---

## 📖 推荐资料

* Triton 官方 examples（attention / matmul）

---

# 📅 第 5 周：kernel fusion（性能关键）

## 🎯 目标

* 理解 fusion 为什么重要

---

## 📚 学习重点

* memory-bound vs compute-bound
* kernel launch overhead

---

## 🧪 实践

实现：

* fused softmax + dropout
* fused add + relu

---

## 📖 推荐资料

* 深度学习优化博客（fusion）

---

# 📅 第 6 周：JIT 编译机制（核心开始）

## 🎯 目标

* 理解 Triton 编译流程

---

## 📚 学习重点

Triton pipeline：

* Python → TTIR → TTGIR → LLVM IR → PTX

涉及：

* LLVM
* PTX

---

## 🧪 实践

开启 debug：

```bash
TRITON_DEBUG=1
```

观察：

* IR 输出
* kernel 编译过程

---

## 📖 推荐资料

* Triton 源码（ir/ codegen/）

---

# 📅 第 7 周：IR 深入理解

## 🎯 目标

* 看懂 TTIR / TTGIR

---

## 📚 学习重点

* SSA（静态单赋值）
* vectorization 在 IR 中如何体现
* memory layout 表达

---

## 🧪 实践

* 对一个 kernel：

  * 看 Python
  * 对比 IR

---

## 📖 推荐资料

* LLVM IR tutorial（非常重要）

---

# 📅 第 8 周：LLVM & PTX

## 🎯 目标

* 理解最终生成代码

---

## 📚 学习重点

* LLVM IR 基本语法
* PTX 指令（load/store）

---

## 🧪 实践

工具：

```bash
nvdisasm
```

分析：

* memory access pattern
* 指令数量

---

## 📖 推荐资料

* LLVM 官方 tutorial
* PTX ISA 文档

---

# 📅 第 9 周：自动调优 & 性能工程

## 🎯 目标

* 能优化到接近极限性能

---

## 📚 学习重点

* autotune
* occupancy vs register tradeoff

---

## 🧪 实践

```python
@triton.autotune(configs=[...])
```

实验：

* 多种 BLOCK_SIZE
* num_warps

---

## 📖 推荐资料

* Triton autotune 文档

---

# 📅 第 10 周：实战项目（非常关键）

## 🎯 目标

* 做一个“像样的工程”

---

## 🧪 项目建议（选一个）

### ⭐ 推荐项目

* FlashAttention（简化版）
* 高性能 matmul
* 自定义 optimizer kernel

---

## 📚 学习重点

* end-to-end 优化
* profiling

---

## 📖 推荐资料

* FlashAttention 论文 + Triton 实现

---

# 🧠 学习方法建议（非常重要）

## 1. 一定要做对比

每个 kernel 都要：

* Triton vs PyTorch
* Triton vs CUDA（可选）

---

## 2. 一定要看 IR（这是分水岭）

👉 不看 IR = 永远停留在“会写代码”
👉 看 IR = 理解“为什么快”

---

## 3. 性能优先思维

问自己：

* 是 compute-bound 还是 memory-bound？
* bottleneck 在哪里？

---

# 🚀 最终能力目标

完成这 10 周后，你应该能：

✅ 写高性能 Triton kernel
✅ 看懂 Triton 编译流程
✅ 分析 GPU 性能瓶颈
✅ 做 kernel-level 优化
