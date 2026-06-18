---
title: DVM架构解析：Kernel对象的设计与实现
date: 2026-05-19 16:33:04
tags:
---

# DVM架构解析：Kernel对象的设计与实现

## 前言

Kernel是DVM中最核心的对象，它承载了从用户构图到最终执行的全部逻辑。理解Kernel的设计与实现，是深入掌握DVM的关键。本文将从架构设计、类层次结构、核心实现等角度，全面解析DVM的Kernel对象。

---

## 一、Kernel架构概览

### 1.1 Kernel的核心职责

Kernel对象在DVM中承担以下职责：

```
┌─────────────────────────────────────────────────────────┐
│                    Kernel核心职责                        │
├─────────────────────────────────────────────────────────┤
│  1. 构图管理：管理元算子对象（NDObject）的生命周期        │
│  2. Shape推导：Normalize阶段进行Shape正则化             │
│  3. 代码生成：CodeGen阶段生成字节码指令                  │
│  4. 执行管理：Launch阶段下发执行到Device                 │
│  5. 内存管理：管理Workspace和UB内存分配                  │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Kernel类型体系

DVM定义了多种Kernel类型，形成清晰的层次结构：

```
VKernel (抽象基类)
│
├── VectorKernel (Vector计算)
│   └── 用于元素级操作、Reduce等
│
├── CubeKernel (Cube计算)
│   └── 用于MatMul、GMM等矩阵运算
│
├── MixKernel (Cube+Vector融合)
│   └── 用于MatMul后接Vector操作
│
├── ParallelKernel (并行堆叠)
│   └── 多个Kernel并行执行
│
├── SequenceKernel (顺序堆叠)
│   └── 多个Kernel顺序执行
│
├── SplitKernel (自动拆分)
│   └── 根据Pattern自动拆分子Kernel
│
└── EagerKernel (动态图)
    └── 动态图模式自动堆叠
```

---

## 二、VKernel基类设计

### 2.1 类定义

```cpp
class VKernel {
 public:
  VKernel(KernelType ktype, uint32_t flags);
  virtual ~VKernel();

  // 核心虚函数
  virtual void Append(NDObject *obj);           // 添加元算子
  virtual void Normalize();                      // Shape推导
  virtual void CodeGenR(...);                    // 代码生成（带重定位）
  virtual int Launch(void *stream);              // 下发执行
  virtual uint64_t CodeGen();                    // 代码生成
  virtual void Dump(...);                        // Dump计算图
  virtual void Clone(...);                       // 克隆Kernel

  // 辅助方法
  std::string &DumpGraph();                      // 获取计算图字符串
  virtual std::string &DisAssemble();            // 反汇编字节码
  KernelType KType() const;                      // 获取Kernel类型
  uint32_t Flags() const;                        // 获取Kernel标志
  bool IsSplit() const;                          // 是否为Split类型
  bool IsDynamic() const;                        // 是否为动态Shape

 protected:
  KernelType ktype_;                             // Kernel类型
  uint32_t flags_;                               // Kernel标志
  Code code_;                                    // 字节码对象
  size_t pre_ws_size_;                           // Workspace大小
  void *pre_ws_mem_;                             // Workspace内存
  MsprofHelper *msprof_;                         // Profiling辅助
  const char *op_name_;                          // 算子名称
  const char *op_fullname_;                      // 算子全名
  IdleCleanWrap *idle_clean_wrap_;               // 空闲清理包装
  std::string dump_str_;                         // Dump字符串缓存
};
```

### 2.2 核心成员解析

#### Code对象

`code_`是Kernel最核心的成员，存储生成的字节码：

```cpp
class Code : public CodeWrap {
 public:
  enum { kTargetVec = 0, kTargetCube, kTargetMix };
  
  uint8_t *data_;           // 字节码数据指针
  size_t data_size_;        // 字节码大小
  uint32_t block_dim_;      // 并行Block数
  int target_;              // 目标类型
  
  // 核心方法
  void Alloc(size_t size);  // 分配字节码内存
  void Launch(void *workspace, void *stream);  // 执行字节码
  void DisAssemble(std::ostringstream &oss);   // 反汇编
};
```

#### KernelType枚举

```cpp
enum KernelType {
  kVector = 0,    // 纯Vector计算
  kCube,          // 纯Cube计算（MatMul）
  kMix,           // Cube+Vector融合
  kParallel,      // 并行堆叠
  kSequence,      // 顺序堆叠
  kSplit,         // 自动拆分
  kEager,         // 动态图模式
};
```

#### KernelFlag枚举

```cpp
enum KernelFlag {
  kDynamic = 0x1,      // 动态Shape标志
  kUnifyWS = 0x2,      // 统一Workspace标志
  kSpeculate = 0x4,    // 投机执行标志
  kPrivate1 = 1u << 30,
  kPrivate2 = 1u << 31,
};
```

### 2.3 核心方法实现

#### Normalize方法

```cpp
void VKernel::Normalize() {
  pre_ws_size_ = CodeGen();  // 执行代码生成并获取Workspace大小
}
```

#### CodeGenR方法（带重定位）

```cpp
void VKernel::CodeGenR(const RelocEntry *relocs, size_t reloc_size, 
                       WsAllocator *ws_alloc) {
  // 1. 重定位地址
  auto reloc = relocs;
  for (size_t i = 0; i < reloc_size; ++i, ++reloc) {
    static_cast<NDAccess *>(reloc->io)->addr_.Reloc(reloc->addr);
  }
  
  // 2. Profiling处理
  if (g_system.enable_profile_) {
    // 初始化或更新Profiling信息
    ...
  }
  
  // 3. 分配Workspace
  if (ws_alloc) {
    pre_ws_mem_ = ws_alloc->Alloc(pre_ws_size_);
  }
}
```

#### Launch方法

```cpp
int VKernel::Launch(void *stream) {
  // 1. 重定位绑定
  code_.RelocBinds(pre_ws_mem_);
  
  // 2. 执行字节码
  if (likely(!g_system.enable_profile_ || msprof_ == nullptr)) {
    return code_.Launch(pre_ws_mem_, stream);
  } else {
    // 带Profiling的执行
    _MsprofLaunchGuard guard(code_, msprof_);
    return code_.Launch(pre_ws_mem_, stream);
  }
}
```

---

## 三、VectorKernel实现

### 3.1 类定义

```cpp
class VectorKernel : public VKernel {
 public:
  explicit VectorKernel(KernelType ktype, uint32_t flags);
  
  void Dump(std::ostringstream &oss, const std::string &indent) override;
  
  // 核心方法
  void BuildDomain();        // 构建计算域
  void PrepareTiling();      // 准备Tiling策略
  
  // 代码生成
  uint8_t *DoCodeGen(uint64_t core_limit, uint8_t *code_ptr, 
                     uint64_t code_reserve);
  uint64_t DoCodeGen(uint64_t core_limit);
  uint64_t DoCodeGenInner(uint64_t core_limit);
  
  // Tiling相关
  void Shard(const ShardParam &sp);
  void SetTile(int start, int end, int64_t num, int64_t factor);
  
  // 辅助方法
  uint64_t ReserveCodeSize() const;
  uint32_t CompactBlockDim(uint64_t core_limit);
  NDAccess *FindInplaceStore(...);
  void CollectIdle(std::vector<NDObject *> &cleans);
  void ProcessIdle();

 protected:
  std::vector<NDObject *> objects_;      // 元算子对象列表
  std::vector<NDObject *> static_ops_;   // 静态算子列表
  Domain *dom_;                          // 计算域
  uint64_t tile_num_;                    // Tile数量
  uint64_t tile_size_;                   // Tile大小
  uint64_t xbuf_size_;                   // UB缓冲区大小
  int max_type_;                         // 最大数据类型
  int min_type_;                         // 最小数据类型
  uint64_t lead_align_;                  // 前导对齐
  CommOp *comm_op_;                      // 通信算子
  const ShardParam *shard_;              // 分片参数
  std::vector<DimTile> tiles_;           // Tile配置
};
```

### 3.2 核心实现

#### DoCodeGen方法

```cpp
uint64_t VectorKernel::DoCodeGen(uint64_t core_limit) {
  // 1. 准备Tiling
  PrepareTiling();
  
  // 2. 处理空Kernel
  if (unlikely(!tile_size_)) {
    ProcessIdle();
    return 0;
  }
  
  // 3. 执行内部代码生成
  return DoCodeGenInner(core_limit);
}

uint64_t VectorKernel::DoCodeGenInner(uint64_t core_limit) {
  // 1. 预留代码空间
  auto code_reserve = ReserveCodeSize();
  code_.Alloc(code_reserve + code_.HeadSize());
  
  // 2. 生成代码
  auto code_end = DoCodeGen(core_limit, code_.data_ + code_.HeadSize(), 
                            code_reserve);
  code_.data_size_ = code_end - code_.data_;
  
  // 3. 更新Block维度
  if (auto visit = GetVisitor<RedVisitCoder>(); visit != nullptr) {
    code_.block_dim_ = CeilDiv<uint32_t>(visit->block_num_, 2);
    code_.UpdateVE(visit);
    return visit->ws_size_;
  }
  
  code_.block_dim_ = CompactBlockDim(core_limit);
  code_.UpdateV(tile_num_);
  return 0;
}
```

#### PrepareTiling方法

```cpp
void VectorKernel::PrepareTiling() {
  // 1. 构建计算域
  BuildDomain();
  
  // 2. 计算Tile参数
  auto &dims = dom_->nd_.dims();
  tile_num_ = 1;
  tile_size_ = 1;
  
  for (size_t i = 0; i < dims.size(); ++i) {
    tile_num_ *= dims[i];
  }
  
  // 3. 根据数据类型计算Tile大小
  tile_size_ = CalculateTileSize(max_type_);
  xbuf_size_ = tile_size_ * ITEM_SIZE[max_type_];
}
```

#### ReserveCodeSize方法

```cpp
uint64_t VectorKernel::ReserveCodeSize() const {
  // 基础大小 + 每个算子的指令大小
  auto res = SIMD_BLOCK_SIZE + objects_.size() * V_INSN_SIZE_MAX;
  
  // 通信算子额外空间
  if (comm_op_) {
    res += comm_op_->CodeReserve();
  }
  
  // 512字节对齐
  return (res + 511ul) & ~511ul;
}
```

---

## 四、CubeKernel实现

### 4.1 类定义

```cpp
class CubeKernel : public VKernel {
 public:
  CubeKernel(KernelType ktype, uint32_t flags);
  ~CubeKernel() override;
  
  void Append(NDObject *obj) override;
  void Dump(std::ostringstream &oss, const std::string &indent) override;
  void Clone(VKernel *base, CloneHelper &helper) override;
  
  uint64_t CodeGen() override;
  uint8_t *DoCodeGen(uint8_t *code_ptr, uint64_t core_limit);

 protected:
  CubeOp *cube_op_;       // Cube算子对象
  bool reload_rhs_;       // 是否重新加载右矩阵
  Tuner *tuner_;          // 调优器
};
```

### 4.2 核心实现

#### Append方法

```cpp
void CubeKernel::Append(NDObject *obj) {
  if (obj->IsCube()) {
    cube_op_ = static_cast<CubeOp *>(obj);
    
    // 处理左右矩阵相同的情况
    if (cube_op_->lhs_ == cube_op_->rhs_) {
      auto lhs = static_cast<NDAccess *>(cube_op_->lhs_);
      cube_op_->rhs_ = new NDLoad(nullptr, lhs->shape_ref_, lhs->type_id_);
      reload_rhs_ = true;
    }
  } else if (obj->IsStore() && obj->lhs_ == cube_op_) {
    cube_op_->output_ = static_cast<NDAccess *>(obj);
  }
}
```

#### DoCodeGen方法

```cpp
uint8_t *CubeKernel::DoCodeGen(uint8_t *code_ptr, uint64_t core_limit) {
  ASSERT(cube_op_->output_ != nullptr);
  
  // 1. 生成Cube指令
  vCubeOp *cube_code = reinterpret_cast<vCubeOp *>(code_ptr);
  cube_op_->CodeGen(cube_code, tuner_);
  
  // 2. 绑定地址
  static_cast<NDAccess *>(cube_op_->lhs_)->addr_.Update(&cube_code->gm_a);
  static_cast<NDAccess *>(cube_op_->rhs_)->addr_.Update(&cube_code->gm_b);
  
  if (cube_code->flags & V_CUBE_FLAG_WITH_BIAS) {
    static_cast<NDAccess *>(cube_op_->bias_)->addr_.Update(&cube_code->gm_bias);
  }
  
  cube_op_->output_->addr_.Update(&cube_code->gm_c);
  
  // 3. 设置Block维度
  code_.block_dim_ = std::min(cube_op_->block_dim_, core_limit);
  
  return code_ptr + sizeof(vCubeOp);
}
```

---

## 五、MixKernel实现

### 5.1 设计思想

MixKernel实现Cube计算与后续Vector计算的融合：

```
传统方式：
MatMul → 写回GM → 读入UB → Bias → Scale → Store
         (多次GM访问，性能差)

Mix融合：
MatMul → [L1 Cache] → Bias → Scale → Store
         (Cube和Vector在Tile粒度Overlap)
```

### 5.2 类定义

```cpp
class MixKernelBase : public CubeKernel {
 public:
  MixKernelBase(KernelType ktype, uint32_t flags);
  ~MixKernelBase() override;
  
  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;

 protected:
  VKernel *post_fusion_;              // 后融合Vector Kernel
  NDLoad *sload_;                     // 从Cube加载的伪Load
  std::vector<std::pair<NDLoad*, NDAccess*>> reloads_;  // 重载列表
};
```

### 5.3 Append方法实现

```cpp
void MixKernelBase::Append(NDObject *obj) {
  constexpr uint32_t LOAD_VEC_USED = 0;
  constexpr uint32_t LOAD_PENDING = 1;
  constexpr uint32_t LOAD_CUBE_USED = 2;
  
  if (obj->IsLoad()) {
    obj->xbuf_ = LOAD_PENDING;
  } else if (obj->IsCube()) {
    // 处理Cube算子
    EXCEPTION_IF(cube_op_ != nullptr, "only one cube op in mix-kernel");
    CubeKernel::Append(obj);
    obj->lhs_->xbuf_ = LOAD_CUBE_USED;
    obj->rhs_->xbuf_ = LOAD_CUBE_USED;
  } else if (obj->IsStore() && obj->lhs_ == cube_op_) {
    cube_op_->output_ = static_cast<NDAccess *>(obj);
  } else {
    // 处理后融合Vector算子
    if (post_fusion_ == nullptr) {
      post_fusion_ = IsDynamic() ? new VKernelD() : new VKernelS();
    }
    
    // 替换Cube输出为伪Load
    obj->ForInput([this](NDObject *&op) {
      if (op == cube_op_) {
        if (sload_ == nullptr) {
          sload_ = new NDLoad(nullptr, cube_op_->shape_ref_, cube_op_->type_id_);
          sload_->flags_ |= OBJ_FLAG_LOAD_FROM_CUBE;
          post_fusion_->Append(sload_);
        }
        op = sload_;
      } else if (op->IsLoad() && op->xbuf_ != LOAD_VEC_USED) {
        // 处理其他Load
        ...
      }
    });
    
    post_fusion_->Append(obj);
  }
}
```

---

## 六、堆叠Kernel实现

### 6.1 ParallelKernel

并行堆叠多个子Kernel到不同核心执行：

```cpp
class ParallelKernel : public VKernel {
 public:
  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  int Launch(void *stream) override;

 private:
  std::vector<VKernel *> sub_kernels_;    // 子Kernel列表
  std::vector<uint32_t> core_limits_;     // 核心数限制
};
```

**使用场景**：
```cpp
// 两个独立计算并行执行
k.ParallelAdd(dvm::kVector);
// ... Vector计算 ...

k.ParallelAdd(dvm::kCube);
// ... Cube计算 ...
```

### 6.2 SequenceKernel

顺序堆叠多个子Kernel：

```cpp
class SequenceKernel : public VKernel {
 public:
  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;
  int Launch(void *stream) override;

 private:
  std::vector<VKernel *> sub_kernels_;    // 子Kernel列表
  std::vector<NDAccess *> intermediates_; // 中间结果
};
```

**特点**：
- 自动插入Load/Store处理中间结果
- 降低运行时调度开销

### 6.3 SplitKernel

自动拆分复杂计算图：

```cpp
class SplitKernel : public VKernel {
 public:
  void Append(NDObject *obj) override;
  uint64_t CodeGen() override;

 private:
  void SplitGraph();                      // 拆分计算图
  std::vector<VKernel *> sub_kernels_;    // 拆分后的子Kernel
};
```

**拆分规则**：
1. Cube算子单独拆分
2. 通信算子单独拆分
3. 相邻Vector算子合并

### 6.4 EagerKernel

动态图模式的自动堆叠：

```cpp
class EagerKernel : public SplitKernel {
 public:
  void Clear();    // 清理状态，准备下一次构图
};
```

**特点**：
- 每次执行后可重新构图
- 支持动态图场景

---

## 七、Kernel生命周期

### 7.1 静态Shape流程

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Reset   │ -> │  Append  │ -> │Normalize │ -> │ CodeGen  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                      │
                                                      v
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  完成    │ <- │ 多次执行  │ <- │  Launch  │ <- │ 缓存代码  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 7.2 动态Shape流程

```
┌──────────┐    ┌──────────┐
│  Reset   │ -> │  Append  │
└──────────┘    └──────────┘
                      │
                      v
              ┌──────────────┐
              │  首次执行    │ <─────────────┐
              └──────────────┘               │
                      │                       │
                      v                       │
              ┌──────────────┐               │
              │  Normalize   │               │
              └──────────────┘               │
                      │                       │
                      v                       │
              ┌──────────────┐               │
              │   CodeGen    │               │
              └──────────────┘               │
                      │                       │
                      v                       │
              ┌──────────────┐               │
              │   Launch     │               │
              └──────────────┘               │
                      │                       │
                      v                       │
              ┌──────────────┐               │
              │ 更新Shape    │ ──────────────┘
              └──────────────┘
```

---

## 八、内存管理

### 8.1 Workspace管理

```cpp
class WsAllocator {
 public:
  virtual void *Alloc(size_t size) = 0;
};

// Device端实现
class DevRunner : public KernelRunner {
  void *Alloc(size_t size) override {
    void *ws = nullptr;
    if (size > 0) {
      aclrtMalloc(&ws, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
      dev_mem_.push_back(ws);
    }
    return ws;
  }
};
```

### 8.2 UB内存管理

VectorKernel通过xbuf管理UB内存：

```cpp
// xbuf分配策略
for (auto op : objects_) {
  if (op->IsLoad()) {
    op->xbuf_ = static_xbuf_;
  } else {
    op->lhs_->xbuf_ = static_xbuf_;
  }
  
  // 根据数据类型分配空间
  if (op->type_id_ != max_type_) {
    static_xbuf_ += tile_size_ * ITEM_SIZE[op->type_id_];
  } else {
    static_xbuf_ += xbuf_size_;
  }
}
```

---

## 九、总结

本文详细解析了DVM Kernel对象的设计与实现：

1. **VKernel基类**：定义了Kernel的核心接口和生命周期
2. **VectorKernel**：实现Vector计算的代码生成和执行
3. **CubeKernel**：实现MatMul等矩阵运算
4. **MixKernel**：实现Cube和Vector的融合执行
5. **堆叠Kernel**：Parallel、Sequence、Split、Eager实现多Kernel组合
6. **内存管理**：Workspace和UB内存的高效管理

理解Kernel的设计，是深入DVM源码的关键一步。下一篇将详细介绍元算子系统的实现。

---

## 参考资料

- [kernel.h](../src/kernel.h)
- [kernel.cc](../src/kernel.cc)
- [xkernel.h](../src/xkernel.h)
- [xkernel.cc](../src/xkernel.cc)
