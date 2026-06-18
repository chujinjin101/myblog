---
title: vllm_learning_plan
date: 2026-05-21 17:31:28
tags:
---

# vLLM 项目学习计划

> 本计划面向希望深入理解 vLLM 推理引擎的开发者，从入门到精通分为 **8 个阶段**，预计总学习时间 **10-12 周**（每天 2-3 小时）。

---

## 阶段一：项目概览与环境搭建（第 1 周，约 7 小时）

### 学习目标
- 理解 vLLM 项目的定位、核心特性和应用场景
- 搭建开发环境，能运行基本推理示例

### 知识点
| 知识点 | 说明 | 推荐时间 |
|--------|------|----------|
| 项目定位与核心特性 | PagedAttention、连续批处理、量化、推测解码等 | 1h |
| 仓库结构概览 | 顶层目录、核心模块划分、双引擎架构 | 1h |
| 开发环境搭建 | uv + Python 3.12 + PyTorch + vLLM 源码安装 | 2h |
| 快速上手推理 | 离线推理 LLM 类 + OpenAI 兼容服务 | 2h |
| 贡献流程 | AGENTS.md、CONTRIBUTING.md、pre-commit | 1h |

### 推荐操作
1. 阅读 [README.md](file:///home/cjj/vllm/README.md)，了解项目全貌
2. 浏览仓库目录结构，建立整体认知
3. 按以下步骤搭建环境：
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -r requirements/lint.txt
   pre-commit install
   VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
   ```
4. 运行基础示例：
   ```bash
   # 离线推理
   .venv/bin/python examples/basic/offline_inference/generate.py
   # 在线服务
   .venv/bin/python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m
   ```
5. 阅读 [docs/getting_started/quickstart.md](file:///home/cjj/vllm/docs/getting_started/quickstart.md)

### 关键文件
- [README.md](file:///home/cjj/vllm/README.md)
- [AGENTS.md](file:///home/cjj/vllm/AGENTS.md)
- [CONTRIBUTING.md](file:///home/cjj/vllm/CONTRIBUTING.md)
- [examples/basic/](file:///home/cjj/vllm/examples/basic)
- [docs/getting_started/](file:///home/cjj/vllm/docs/getting_started)

---

## 阶段二：架构总览与核心概念（第 2 周，约 10 小时）

### 学习目标
- 理解 vLLM 的整体架构分层
- 掌握请求从提交到返回的完整数据流
- 理解 V1 引擎的设计动机和架构

### 知识点
| 知识点 | 说明 | 推荐时间 |
|--------|------|----------|
| 架构分层 | 入口层→引擎层→调度层→执行层→Worker层→模型层→内核层→平台层 | 2h |
| 入口点体系 | CLI / Python API (LLM) / HTTP API (OpenAI) / gRPC | 1h |
| V1 引擎架构 | AsyncLLM + EngineCore + ZMQ 通信 | 2h |
| 请求完整生命周期 | 从 generate() 到 RequestOutput 的端到端流程 | 2h |
| 多进程架构 | API 进程 + EngineCore 后台进程 + Worker 进程 | 1h |
| 配置系统 | VllmConfig + 28 个子配置模块 | 2h |

### 推荐操作
1. **精读** [docs/design/arch_overview.md](file:///home/cjj/vllm/docs/design/arch_overview.md)（最重要的一篇文档）
2. **精读** [docs/design/multiprocessing.md](file:///home/cjj/vllm/docs/design/multiprocessing.md)
3. 阅读入口点代码：
   - [vllm/entrypoints/llm.py](file:///home/cjj/vllm/vllm/entrypoints/llm.py) — LLM 类
   - [vllm/entrypoints/openai/api_server.py](file:///home/cjj/vllm/vllm/entrypoints/openai/api_server.py) — API 服务器
   - [vllm/entrypoints/cli/main.py](file:///home/cjj/vllm/vllm/entrypoints/cli/main.py) — CLI 入口
4. 阅读 V1 引擎核心代码：
   - [vllm/v1/engine/async_llm.py](file:///home/cjj/vllm/vllm/v1/engine/async_llm.py) — AsyncLLM
   - [vllm/v1/engine/core.py](file:///home/cjj/vllm/vllm/v1/engine/core.py) — EngineCore
   - [vllm/v1/engine/core_client.py](file:///home/cjj/vllm/vllm/v1/engine/core_client.py) — 通信客户端
5. 画一张请求流转图，标注关键类和方法
6. 阅读 [docs/configuration/README.md](file:///home/cjj/vllm/docs/configuration/README.md) 和 [docs/configuration/engine_args.md](file:///home/cjj/vllm/docs/configuration/engine_args.md)

### 关键文件
- [vllm/v1/engine/async_llm.py](file:///home/cjj/vllm/vllm/v1/engine/async_llm.py)
- [vllm/v1/engine/core.py](file:///home/cjj/vllm/vllm/v1/engine/core.py)
- [vllm/v1/engine/core_client.py](file:///home/cjj/vllm/vllm/v1/engine/core_client.py)
- [vllm/v1/engine/input_processor.py](file:///home/cjj/vllm/vllm/v1/engine/input_processor.py)
- [vllm/v1/engine/output_processor.py](file:///home/cjj/vllm/vllm/v1/engine/output_processor.py)
- [vllm/config/vllm.py](file:///home/cjj/vllm/vllm/config/vllm.py)

---

## 阶段三：PagedAttention 与 KV Cache 管理（第 3-4 周，约 16 小时）

### 学习目标
- 深入理解 PagedAttention 的核心思想和实现
- 掌握 KV Cache 的块分配、释放、前缀缓存机制
- 理解调度器如何与 KV Cache 管理器协作

### 知识点
| 知识点 | 说明 | 推荐时间 |
|--------|------|----------|
| PagedAttention 原理 | 虚拟内存分页类比、块式 KV cache、内存碎片消除 | 3h |
| KV Cache 内存布局 | k_cache/v_cache 的 5D 张量形状 | 2h |
| BlockPool 与块管理 | 空闲块队列(LRU)、引用计数、哈希查找 | 2h |
| 前缀缓存 (APC) | 块哈希、最长前缀匹配、共享块引用计数 | 2h |
| KV Cache 类型 | FullAttention/SlidingWindow/MLA/Mamba/ChunkedLocal | 2h |
| KV Connector | 分布式 KV 传输、Prefill/Decode 分离、异步加载 | 2h |
| 抢占与驱逐 | KV cache 不足时的抢占策略、请求重计算 | 1h |
| 混合 KV Cache 管理 | KVCacheCoordinator + SingleTypeKVCacheManager | 2h |

### 推荐操作
1. **精读** [docs/design/paged_attention.md](file:///home/cjj/vllm/docs/design/paged_attention.md)
2. **精读** [docs/design/prefix_caching.md](file:///home/cjj/vllm/docs/design/prefix_caching.md)
3. **精读** [docs/design/hybrid_kv_cache_manager.md](file:///home/cjj/vllm/docs/design/hybrid_kv_cache_manager.md)
4. 阅读 KV Cache 管理核心代码：
   - [vllm/v1/core/kv_cache_manager.py](file:///home/cjj/vllm/vllm/v1/core/kv_cache_manager.py)
   - [vllm/v1/core/block_pool.py](file:///home/cjj/vllm/vllm/v1/core/block_pool.py)
   - [vllm/v1/core/kv_cache_utils.py](file:///home/cjj/vllm/vllm/v1/core/kv_cache_utils.py)
5. 阅读 CUDA 内核实现（选读）：
   - [csrc/attention/paged_attention_v2.cu](file:///home/cjj/vllm/csrc/attention/paged_attention_v2.cu)
6. 运行前缀缓存示例：
   ```bash
   .venv/bin/python examples/features/automatic_prefix_caching/
   ```
7. 阅读 KV Connector 代码：
   - [vllm/v1/worker/gpu/kv_connector.py](file:///home/cjj/vllm/vllm/v1/worker/gpu/kv_connector.py)
   - [vllm/distributed/kv_transfer/](file:///home/cjj/vllm/vllm/distributed/kv_transfer)
8. 阅读混合 KV Cache 文档：[docs/design/hybrid_kv_cache_manager.md](file:///home/cjj/vllm/docs/design/hybrid_kv_cache_manager.md)

### 关键文件
- [vllm/v1/core/kv_cache_manager.py](file:///home/cjj/vllm/vllm/v1/core/kv_cache_manager.py)
- [vllm/v1/core/block_pool.py](file:///home/cjj/vllm/vllm/v1/core/block_pool.py)
- [vllm/v1/core/kv_cache_utils.py](file:///home/cjj/vllm/vllm/v1/core/kv_cache_utils.py)
- [vllm/v1/kv_cache_interface.py](file:///home/cjj/vllm/vllm/v1/kv_cache_interface.py)
- [csrc/attention/](file:///home/cjj/vllm/csrc/attention)

---

## 阶段四：调度器与请求管理（第 4-5 周，约 10 小时）

### 学习目标
- 理解 V1 统一调度算法（基于 computed tokens 追赶模型）
- 掌握请求状态机（WAITING → RUNNING → FINISHED）
- 理解调度约束和抢占机制

### 知识点
| 知识点 | 说明 | 推荐时间 |
|--------|------|----------|
| 统一调度算法 | 无传统 prefill/decode 分离，基于 num_computed_tokens 追赶 | 3h |
| 请求状态机 | WAITING/RUNNING/PREEMPTED/FINISHED/SWAPPED 等状态 | 2h |
| 调度约束 | max_num_running_reqs, max_num_scheduled_tokens, KV 块可用性 | 2h |
| 抢占策略 | 最低优先级抢占、全量重计算 vs 增量重计算 | 1h |
| 请求队列管理 | FCFS/PRIORITY 策略、skipped_waiting 机制 | 1h |
| 输出更新 | update_from_output() 处理采样结果、停止条件、投机解码 | 1h |

### 推荐操作
1. **精读** [vllm/v1/core/sched/scheduler.py](file:///home/cjj/vllm/vllm/v1/core/sched/scheduler.py)（核心文件，约 1600 行）
   - 重点阅读 `schedule()` 方法（L329-L922）
   - 重点阅读 `update_from_output()` 方法（L1283-L1603）
2. 阅读请求定义：[vllm/v1/core/req/](file:///home/cjj/vllm/vllm/v1/core/req)
3. 阅读调度器配置：[vllm/config/scheduler.py](file:///home/cjj/vllm/vllm/config/scheduler.py)
4. 运行调度器相关测试：
   ```bash
   .venv/bin/python -m pytest tests/v1/core/ -v -k scheduler
   ```
5. 尝试修改调度参数（如 max_num_running_reqs），观察行为变化

### 关键文件
- [vllm/v1/core/sched/scheduler.py](file:///home/cjj/vllm/vllm/v1/core/sched/scheduler.py)
- [vllm/v1/core/sched/async_scheduler.py](file:///home/cjj/vllm/vllm/v1/core/sched/async_scheduler.py)
- [vllm/v1/core/req/](file:///home/cjj/vllm/vllm/v1/core/req)
- [vllm/config/scheduler.py](file:///home/cjj/vllm/vllm/config/scheduler.py)

---

## 阶段五：模型执行与注意力后端（第 5-7 周，约 18 小时）

### 学习目标
- 理解 GPU ModelRunner 的推理循环
- 掌握注意力后端的可插拔架构
- 理解 CUDA Graph 优化机制
- 了解模型注册与加载流程

### 知识点
| 知识点 | 说明 | 推荐时间 |
|--------|------|----------|
| GPU ModelRunner | 推理循环、输入批处理、CUDA Graph 捕获 | 3h |
| 注意力后端架构 | AttentionBackend 抽象 + FlashAttn/FlashInfer/Triton 等 | 3h |
| FlashAttention | 最主流的注意力后端，FlashAttention 2/3 | 2h |
| FlashInfer | 高性能注意力库，支持 batch prefill/decode | 2h |
| MLA 注意力 | DeepSeek 的 Multi-head Latent Attention | 1h |
| CUDA Graphs | 消除 CPU 开销、图捕获与重放 | 2h |
| 模型注册与加载 | ModelRegistry、模型权重加载器 | 2h |
| torch.compile 集成 | 分段编译、编译优化级别 | 2h |
| InputBatch | 批处理数据结构、block_table 管理 | 1h |

### 推荐操作
1. **精读** [docs/design/attention_backends.md](file:///home/cjj/vllm/docs/design/attention_backends.md)
2. **精读** [docs/design/cuda_graphs.md](file:///home/cjj/vllm/docs/design/cuda_graphs.md)
3. **精读** [docs/design/model_runner_v2.md](file:///home/cjj/vllm/docs/design/model_runner_v2.md)
4. **精读** [docs/design/torch_compile.md](file:///home/cjj/vllm/docs/design/torch_compile.md)
5. 阅读 GPU ModelRunner 代码：
   - [vllm/v1/worker/gpu/model_runner.py](file:///home/cjj/vllm/vllm/v1/worker/gpu/model_runner.py)
   - [vllm/v1/worker/gpu/input_batch.py](file:///home/cjj/vllm/vllm/v1/worker/gpu/input_batch.py)
   - [vllm/v1/worker/gpu/block_table.py](file:///home/cjj/vllm/vllm/v1/worker/gpu/block_table.py)
6. 阅读注意力后端代码：
   - [vllm/v1/attention/backend.py](file:///home/cjj/vllm/vllm/v1/attention/backend.py) — 抽象基类
   - [vllm/v1/attention/backends/flash_attn.py](file:///home/cjj/vllm/vllm/v1/attention/backends/flash_attn.py)
   - [vllm/v1/attention/backends/flashinfer.py](file:///home/cjj/vllm/vllm/v1/attention/backends/flashinfer.py)
7. 阅读模型注册代码：
   - [vllm/model_executor/models/registry.py](file:///home/cjj/vllm/vllm/model_executor/models/registry.py)
8. 阅读一个具体模型实现（如 Llama）：
   - [vllm/model_executor/models/llama.py](file:///home/cjj/vllm/vllm/model_executor/models/llama.py)
9. 阅读 HuggingFace 集成文档：[docs/design/huggingface_integration.md](file:///home/cjj/vllm/docs/design/huggingface_integration.md)

### 关键文件
- [vllm/v1/worker/gpu/model_runner.py](file:///home/cjj/vllm/vllm/v1/worker/gpu/model_runner.py)
- [vllm/v1/attention/backend.py](file:///home/cjj/vllm/vllm/v1/attention/backend.py)
- [vllm/v1/attention/backends/flash_attn.py](file:///home/cjj/vllm/vllm/v1/attention/backends/flash_attn.py)
- [vllm/v1/attention/backends/flashinfer.py](file:///home/cjj/vllm/vllm/v1/attention/backends/flashinfer.py)
- [vllm/model_executor/models/llama.py](file:///home/cjj/vllm/vllm/model_executor/models/llama.py)
- [vllm/model_executor/models/registry.py](file:///home/cjj/vllm/vllm/model_executor/models/registry.py)

---

## 阶段六：采样、推测解码与高级特性（第 7-8 周，约 12 小时）

### 学习目标
- 理解采样管线和 logits 处理流程
- 掌握推测解码的原理和实现
- 了解量化、LoRA、结构化输出等高级特性

### 知识点
| 知识点 | 说明 | 推荐时间 |
|--------|------|----------|
| 采样管线 | logits→惩罚→temperature→top_k/p→采样 | 2h |
| 推测解码原理 | draft-verify 范式、接受/拒绝采样 | 2h |
| EAGLE 推测解码 | 自回归投机、特征注入 | 1h |
| N-gram / 后缀解码 | 无需 draft model 的推测解码 | 1h |
| 量化方案 | FP8/INT8/INT4/GPTQ/AWQ/GGUF/BNB 等 20+ 种 | 2h |
| LoRA | 多 LoRA 适配器管理、动态切换 | 1h |
| 结构化输出 | JSON Schema / Regex 约束、grammar bitmask | 1h |
| 多模态处理 | 图像/音频/视频输入处理管线 | 1h |
| 工具调用 | Function Calling、40+ 工具解析器 | 1h |

### 推荐操作
1. 阅读采样器代码：
   - [vllm/v1/sample/sampler.py](file:///home/cjj/vllm/vllm/v1/sample/sampler.py)
   - [vllm/v1/worker/gpu/sample/sampler.py](file:///home/cjj/vllm/vllm/v1/worker/gpu/sample/sampler.py)
2. 阅读推测解码代码和文档：
   - [docs/features/speculative_decoding/README.md](file:///home/cjj/vllm/docs/features/speculative_decoding/README.md)
   - [vllm/v1/spec_decode/](file:///home/cjj/vllm/vllm/v1/spec_decode)
3. 运行推测解码示例：
   ```bash
   .venv/bin/python examples/features/speculative_decoding/
   ```
4. 阅读量化文档：
   - [docs/features/quantization/README.md](file:///home/cjj/vllm/docs/features/quantization/README.md)
   - [docs/features/quantization/fp8.md](file:///home/cjj/vllm/docs/features/quantization/fp8.md)
5. 阅读 LoRA 文档和代码：
   - [docs/features/lora.md](file:///home/cjj/vllm/docs/features/lora.md)
   - [vllm/lora/](file:///home/cjj/vllm/vllm/lora)
6. 阅读结构化输出文档：
   - [docs/features/structured_outputs.md](file:///home/cjj/vllm/docs/features/structured_outputs.md)
7. 阅读多模态处理文档：
   - [docs/design/mm_processing.md](file:///home/cjj/vllm/docs/design/mm_processing.md)

### 关键文件
- [vllm/v1/sample/sampler.py](file:///home/cjj/vllm/vllm/v1/sample/sampler.py)
- [vllm/v1/spec_decode/](file:///home/cjj/vllm/vllm/v1/spec_decode)
- [vllm/model_executor/layers/quantization/](file:///home/cjj/vllm/vllm/model_executor/layers/quantization)
- [vllm/lora/](file:///home/cjj/vllm/vllm/lora)
- [vllm/v1/structured_output/](file:///home/cjj/vllm/vllm/v1/structured_output)
- [vllm/multimodal/](file:///home/cjj/vllm/vllm/multimodal)

---

## 阶段七：分布式与部署（第 9-10 周，约 12 小时）

### 学习目标
- 理解张量并行(TP)、流水线并行(PP)、数据并行(DP)、专家并行(EP)
- 掌握分布式执行器架构
- 了解生产部署方案

### 知识点
| 知识点 | 说明 | 推荐时间 |
|--------|------|----------|
| 张量并行 (TP) | 模型按层切分到多 GPU | 2h |
| 流水线并行 (PP) | 模型按阶段切分 | 1h |
| 数据并行 (DP) | 请求级数据并行 | 1h |
| 专家并行 (EP) | MoE 模型的专家切分 | 2h |
| 上下文并行 (CP) | 长序列上下文切分 | 1h |
| 执行器架构 | UniProc / MultiProc / Ray 执行器 | 2h |
| 分离式预填充 | Prefill/Decode 分离部署 | 1h |
| Docker/K8s 部署 | 生产环境部署方案 | 1h |
| 性能调优 | 内存节省、优化配置、基准测试 | 1h |

### 推荐操作
1. 阅读并行文档：
   - [docs/serving/parallelism_scaling.md](file:///home/cjj/vllm/docs/serving/parallelism_scaling.md)
   - [docs/serving/data_parallel_deployment.md](file:///home/cjj/vllm/docs/serving/data_parallel_deployment.md)
   - [docs/serving/expert_parallel_deployment.md](file:///home/cjj/vllm/docs/serving/expert_parallel_deployment.md)
2. 阅读执行器代码：
   - [vllm/v1/executor/abstract.py](file:///home/cjj/vllm/vllm/v1/executor/abstract.py)
   - [vllm/v1/executor/uniproc_executor.py](file:///home/cjj/vllm/vllm/v1/executor/uniproc_executor.py)
   - [vllm/v1/executor/multiproc_executor.py](file:///home/cjj/vllm/vllm/v1/executor/multiproc_executor.py)
3. 阅读分布式通信代码：
   - [vllm/distributed/parallel_state.py](file:///home/cjj/vllm/vllm/distributed/parallel_state.py)
   - [vllm/distributed/communication_op.py](file:///home/cjj/vllm/vllm/distributed/communication_op.py)
4. 阅读分离式预填充文档：
   - [docs/features/disagg_prefill.md](file:///home/cjj/vllm/docs/features/disagg_prefill.md)
5. 运行分布式示例（需多 GPU）：
   ```bash
   .venv/bin/python examples/features/torchrun/
   ```
6. 阅读部署文档：
   - [docs/deployment/docker.md](file:///home/cjj/vllm/docs/deployment/docker.md)
   - [docs/deployment/k8s.md](file:///home/cjj/vllm/docs/deployment/k8s.md)
7. 运行基准测试：
   ```bash
   .venv/bin/python -m vllm bench latency --model facebook/opt-125m
   .venv/bin/python -m vllm bench throughput --model facebook/opt-125m
   ```

### 关键文件
- [vllm/v1/executor/](file:///home/cjj/vllm/vllm/v1/executor)
- [vllm/distributed/parallel_state.py](file:///home/cjj/vllm/vllm/distributed/parallel_state.py)
- [vllm/distributed/kv_transfer/](file:///home/cjj/vllm/vllm/distributed/kv_transfer)
- [vllm/config/parallel.py](file:///home/cjj/vllm/vllm/config/parallel.py)

---

## 阶段八：内核优化与深度定制（第 10-12 周，约 14 小时）

### 学习目标
- 理解 CUDA/Triton 内核的设计与优化
- 掌握自定义算子注册机制
- 了解 MoE 融合内核
- 能够为 vLLM 贡献代码

### 知识点
| 知识点 | 说明 | 推荐时间 |
|--------|------|----------|
| CUDA 内核基础 | Paged Attention 内核、采样器内核 | 3h |
| Triton 内核 | Python 层内核、Helion 集成 | 2h |
| 自定义算子注册 | custom_op 机制、torch.library 集成 | 2h |
| 算子融合 | 融合 MoE、融合激活函数、RMSNorm | 2h |
| MoE 内核 | topk 路由、专家计算、Marlin 量化 | 2h |
| 平台抽象层 | CUDA/ROCm/CPU/XPU/TPU 适配 | 1h |
| 插件系统 | LoRA 解析器插件、IO 处理器插件 | 1h |
| 贡献实践 | 添加新模型、编写测试、提交 PR | 1h |

### 推荐操作
1. 阅读 CUDA 内核代码（选读，按兴趣）：
   - [csrc/attention/paged_attention_v2.cu](file:///home/cjj/vllm/csrc/attention/paged_attention_v2.cu)
   - [csrc/moe/](file:///home/cjj/vllm/csrc/moe)
   - [csrc/quantization/](file:///home/cjj/vllm/csrc/quantization)
2. 阅读设计文档：
   - [docs/design/custom_op.md](file:///home/cjj/vllm/docs/design/custom_op.md)
   - [docs/design/fusions.md](file:///home/cjj/vllm/docs/design/fusions.md)
   - [docs/design/fused_moe_modular_kernel.md](file:///home/cjj/vllm/docs/design/fused_moe_modular_kernel.md)
   - [docs/design/plugin_system.md](file:///home/cjj/vllm/docs/design/plugin_system.md)
3. 阅读平台抽象层：
   - [vllm/platforms/](file:///home/cjj/vllm/vllm/platforms)
4. 阅读 MoE 层代码：
   - [vllm/model_executor/layers/fused_moe/](file:///home/cjj/vllm/vllm/model_executor/layers/fused_moe)
5. 尝试贡献：
   - 阅读 [docs/contributing/model/README.md](file:///home/cjj/vllm/docs/contributing/model/README.md)
   - 尝试添加一个简单模型或修复一个小 bug
   - 运行测试验证：
     ```bash
     .venv/bin/python -m pytest tests/v1/ -v -k "test_name"
     pre-commit run --all-files
     ```

### 关键文件
- [csrc/](file:///home/cjj/vllm/csrc)
- [vllm/model_executor/layers/fused_moe/](file:///home/cjj/vllm/vllm/model_executor/layers/fused_moe)
- [vllm/kernels/](file:///home/cjj/vllm/vllm/kernels)
- [vllm/platforms/](file:///home/cjj/vllm/vllm/platforms)
- [vllm/plugins/](file:///home/cjj/vllm/vllm/plugins)

---

## 学习时间总览

| 阶段 | 主题 | 周次 | 预计时间 | 难度 |
|------|------|------|----------|------|
| 一 | 项目概览与环境搭建 | 第 1 周 | 7h | ⭐ |
| 二 | 架构总览与核心概念 | 第 2 周 | 10h | ⭐⭐ |
| 三 | PagedAttention 与 KV Cache | 第 3-4 周 | 16h | ⭐⭐⭐⭐ |
| 四 | 调度器与请求管理 | 第 4-5 周 | 10h | ⭐⭐⭐ |
| 五 | 模型执行与注意力后端 | 第 5-7 周 | 18h | ⭐⭐⭐⭐ |
| 六 | 采样、推测解码与高级特性 | 第 7-8 周 | 12h | ⭐⭐⭐ |
| 七 | 分布式与部署 | 第 9-10 周 | 12h | ⭐⭐⭐ |
| 八 | 内核优化与深度定制 | 第 10-12 周 | 14h | ⭐⭐⭐⭐⭐ |
| **总计** | | **12 周** | **~99h** | |

---

## 学习建议

### 必读文档优先级
1. **必读**（阶段 1-4）：arch_overview.md → paged_attention.md → multiprocessing.md → prefix_caching.md
2. **推荐**（阶段 5-6）：attention_backends.md → cuda_graphs.md → model_runner_v2.md → torch_compile.md
3. **选读**（阶段 7-8）：按兴趣选择量化/分布式/内核相关文档

### 代码阅读技巧
1. **自顶向下**：从入口点开始，沿调用链深入
2. **抓主干**：先理解核心流程，再关注分支逻辑
3. **画图辅助**：用流程图/架构图帮助理解复杂交互
4. **调试跟踪**：用 `pdb` 或 `logging` 跟踪实际执行路径
5. **测试驱动**：阅读测试代码理解组件行为

### 实践建议
1. 每个阶段都运行对应的示例代码
2. 尝试修改参数观察行为变化
3. 阅读测试代码理解预期行为
4. 尝试修复简单 issue 或添加小功能
5. 写学习笔记加深理解

### 前置知识要求
- **必须**：Python、PyTorch 基础、Transformer 架构
- **推荐**：CUDA 编程基础、操作系统虚拟内存概念、分布式系统基础
- **加分**：Triton 编程、GPU 性能优化经验、FastAPI/asyncio
