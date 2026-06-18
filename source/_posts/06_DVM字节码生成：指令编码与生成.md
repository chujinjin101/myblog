---
title: DVM字节码生成：指令编码与生成
date: 2026-05-19 16:33:04
tags:
---

# DVM字节码生成：指令编码与生成

## 前言

字节码生成是DVM编译流程的核心环节，它将计算图转换为可在Device端执行的虚拟机指令。本文将深入解析DVM的字节码生成机制，包括指令集架构、编码格式、代码生成流程等核心内容。

---

## 一、指令集架构

### 1.1 指令分类

DVM的虚拟机指令分为三大类：

```
┌─────────────────────────────────────────────────────────┐
│                  DVM指令分类                             │
├─────────────────────────────────────────────────────────┤
│  访存指令 (vAccInsnID)                                   │
│  - V_LOAD: 从GM加载到UB                                 │
│  - V_STORE: 从UB存储到GM                                │
│  - V_SLOAD: 标量加载                                    │
│  - V_SSTORE: 标量存储                                   │
│  - V_MULTI_LOAD: 多卡加载                               │
│  - V_PINGPONG_LOAD: 双缓冲加载                          │
├─────────────────────────────────────────────────────────┤
│  SIMD指令 (vSimdInsnID)                                 │
│  - V_ADD, V_SUB, V_MUL, V_DIV: 算术运算                 │
│  - V_SQRT, V_EXP, V_LOG: 数学函数                       │
│  - V_CMP: 比较运算                                      │
│  - V_CAST: 类型转换                                     │
│  - V_RSUM_X, V_RMAX_X: 归约运算                         │
├─────────────────────────────────────────────────────────┤
│  访问模式指令 (vVisitID)                                 │
│  - V_VISIT_RED_1/2/3/4: Reduce访问模式                  │
│  - V_VISIT_MIX: Mix访问模式                             │
│  - V_VISIT_REORDER: 重排序访问                          │
└─────────────────────────────────────────────────────────┘
```

### 1.2 指令ID定义

```cpp
enum vAccInsnID {
  V_LOAD = 0,
  V_LOAD_DUMMY,
  V_LOAD_VIEW,
  V_SLOAD,
  V_LOAD_CC,
  V_MULTI_LOAD,
  V_PINGPONG_LOAD,
  V_PINGPONG_PEER_LOAD,
  V_PEER_LOAD,
  V_PEER_LOAD_MIX,
  V_STORE,
  V_STORE_ATOMIC,
  V_STORE_COND,
  V_SSTORE,
  V_SLICE_STORE,
  V_STORE_AG,
  V_STORE_RS,
  V_PEER_STORE,
  V_PEER_STORE_MIX,
  V_ACCESS_NONE,
};

enum vSimdInsnID {
  V_COPY = 0,
  V_COPY_CUBE_TILE,
  V_NOP,
  V_BROADCAST_Y,
  V_BROADCAST_S,
  V_SQRT,
  V_ABS,
  V_LOG,
  V_EXP,
  V_ROUND,
  V_FLOOR,
  V_CEIL,
  V_TRUNC,
  V_ADDS,
  V_MULS,
  V_DIVS,
  V_SDIV,
  V_CMPS,
  V_ADD,
  V_SUB,
  V_MUL,
  V_DIV,
  V_MIN,
  V_MAX,
  V_CMP,
  V_CAST_FP32_TO_FP16,
  V_CAST_FP32_TO_INT32,
  V_RSUM_X,
  V_RSUM_Y,
  V_RMAX_X,
  V_RMAX_Y,
  V_RMIN_X,
  V_RMIN_Y,
  V_RSUM_JOIN,
  V_SEL,
  V_POW,
  V_CLR_PAD,
  V_ELEMENT_ANY,
  V_ONE_HOT,
  // ... 更多指令
};

enum vVisitID {
  V_VISIT_RED_1 = 0,
  V_VISIT_RED_2,
  V_VISIT_RED_3,
  V_VISIT_RED_4,
  V_VISIT_MIX,
  V_VISIT_REORDER,
  V_VISIT_PIPE_SET,
  V_VISIT_PIPE_WAIT,
  V_VISIT_NONE,
};
```

---

## 二、指令编码格式

### 2.1 基础编码结构

DVM指令采用64位定长编码：

```cpp
// 通用指令格式
struct vInsn {
  uint64_t opcode : 8;    // 操作码
  uint64_t xd : 8;        // 目标寄存器
  uint64_t xn : 8;        // 源寄存器1
  uint64_t xm : 8;        // 源寄存器2
  uint64_t reserved : 32; // 保留/扩展
};
```

### 2.2 Load指令编码

```cpp
struct vLoad {
  uint64_t xn : 8;         // UB目标地址
  uint64_t iter_size : 16; // 迭代大小
  uint64_t body_iter : 16; // 主体迭代次数
  uint64_t tile_stride : 8;// Tile步长
  uint64_t pad_size : 8;   // Padding大小
  uint64_t tail_iter : 8;  // 尾部迭代
  uint64_t round_rank : 8; // 广播维度
  uint64_t reserved : 8;
  uint64_t from;           // GM源地址（64位）
};

static inline uint64_t Encode(uint64_t *insn, vSimdInsnID id, const vLoad &op) {
  insn[0] = id | op.xn << 8 | op.iter_size << 16 | op.body_iter << 32 |
            op.tile_stride << 48 | op.pad_size << 56;
  insn[1] = op.tail_iter | op.round_rank << 8;
  insn[2] = op.from;
  return 3;  // 返回指令长度（64位字）
}
```

### 2.3 Store指令编码

```cpp
struct vStore {
  uint64_t xn : 8;         // UB源地址
  uint64_t iter_size : 16; // 迭代大小
  uint64_t body_iter : 16; // 主体迭代次数
  uint64_t tile_stride : 8;// Tile步长
  uint64_t pad_size : 8;   // Padding大小
  uint64_t tail_iter : 8;  // 尾部迭代
  uint64_t round_rank : 8; // 广播维度
  uint64_t reserved : 8;
  uint64_t to;             // GM目标地址
};
```

### 2.4 Binary指令编码

```cpp
struct vBinary {
  uint64_t xd : 8;   // 目标UB地址
  uint64_t xn : 8;   // 源UB地址1
  uint64_t xm : 8;   // 源UB地址2
  uint64_t reserved : 40;
};

static inline uint64_t Encode(uint64_t *insn, vSimdInsnID id, const vBinary &op) {
  insn[0] = id | op.xd << 8 | op.xn << 16 | op.xm << 24;
  return 1;
}
```

### 2.5 Reduce指令编码

```cpp
struct vReduce {
  uint64_t xd : 8;        // 目标UB地址
  uint64_t xn : 8;        // 源UB地址
  uint64_t iter_num : 16; // 迭代次数
  uint64_t simd_width : 8;// SIMD宽度
  uint64_t iter_tail : 8; // 尾部迭代
  uint64_t reserved : 16;
};

static inline uint64_t Encode(uint64_t *insn, vSimdInsnID id, const vReduce &op) {
  insn[0] = id | op.xd << 8 | op.xn << 16 | op.iter_num << 24 |
            op.simd_width << 40 | op.iter_tail << 48;
  return 1;
}
```

### 2.6 MatMul指令编码

```cpp
struct vCubeOp {
  uint64_t flags : 8;      // 标志位
  uint64_t block_dim : 8;  // Block维度
  uint64_t core_loop : 16; // 核心循环
  uint64_t reserved : 32;
  uint64_t gm_a;           // 矩阵A地址
  uint64_t gm_b;           // 矩阵B地址
  uint64_t gm_c;           // 输出地址
  uint64_t gm_bias;        // Bias地址（可选）
  uint64_t gm_group_list;  // 分组列表（GMM）
  // ... 更多参数
};

// 标志位定义
enum vCubeFlag {
  V_CUBE_FLAG_WITH_BIAS = 0x1,
  V_CUBE_FLAG_TRANS_A = 0x2,
  V_CUBE_FLAG_TRANS_B = 0x4,
  V_CUBE_FLAG_GROUPED_LIST = 0x8,
};
```

---

## 三、Code类设计

### 3.1 Code类定义

```cpp
class Code : public CodeWrap {
 public:
  enum { kTargetVec = 0, kTargetCube, kTargetMix };
  
  Code() = default;
  ~Code() override;
  
  // 内存管理
  void Alloc(size_t size);
  void MoveCode(Code &other);
  
  // 入口生成
  static inline uint64_t GenEntry(uint64_t data, uint64_t ktype, uint64_t data_size);
  static inline uint64_t GenEntryV(uint64_t tile_num, uint64_t block_dim, uint64_t data_size);
  static inline uint64_t GenEntryVE(uint64_t visit_id, uint64_t visit_offset, uint64_t data_size);
  static inline uint64_t GenEntryC(uint64_t data_size);
  
  // 头部更新
  void UpdateHead(uint64_t entry);
  void UpdateV(uint64_t tile_num);
  void UpdateVE(const RedVisitCoder *visit);
  
  // 执行
  int Launch(void *workspace, void *stream);
  
  // 调试
  void DisAssemble(std::ostringstream &oss);
  void RelocBinds(void *workspace);
  
  // 数据成员
  uint8_t *data_{nullptr};
  size_t data_size_{0};
  uint32_t block_dim_{1};
  int target_{kTargetVec};
  
  // 重定位信息
  std::vector<RelocAddr *> bind_ops_;
  std::vector<RelocAddr *> bind_wss_;
};
```

### 3.2 入口编码

```cpp
// 入口类型定义
#define V_ENTRY_TYPE_V    0x0000000000000000UL
#define V_ENTRY_TYPE_C    0x4000000000000000UL
#define V_ENTRY_TYPE_VE   0x8000000000000000UL

// Vector入口
static inline uint64_t GenEntryV(uint64_t tile_num, uint64_t block_dim, uint64_t data_size) {
  uint64_t block_tile = CeilDiv<uint64_t>(tile_num, block_dim);
  uint64_t block_tail = block_dim * block_tile - tile_num;
  
  uint64_t data = block_tile << V_ENTRY_V_TILE_BODY_OFFSET | 
                  block_tail << V_ENTRY_V_TILE_TAIL_OFFSET;
  return GenEntry(data, V_ENTRY_TYPE_V, data_size);
}

// Cube/Mix入口
static inline uint64_t GenEntryC(uint64_t data_size) {
  return GenEntry(V_ENTRY_FLAG_CUBE_MIX, V_ENTRY_TYPE_C, data_size);
}

// Reduce访问入口
static inline uint64_t GenEntryVE(uint64_t visit_id, uint64_t visit_offset, uint64_t data_size) {
  uint64_t data = g_system.g_visit_func_offset_[visit_id] << V_ENTRY_VE_VISIT_ID_OFFSET |
                  visit_offset << V_ENTRY_VE_VISIT_OFFSET_OFFSET;
  return GenEntry(data, V_ENTRY_TYPE_VE, data_size);
}
```

### 3.3 内存分配

```cpp
void Code::Alloc(size_t size) {
  // 释放旧内存
  if (data_) {
    std::free(data_);
  }
  
  // 分配新内存（64字节对齐）
  data_ = static_cast<uint8_t *>(std::aligned_alloc(64, size));
  data_size_ = size;
  
  // 初始化头部
  uint64_t *head = reinterpret_cast<uint64_t *>(data_);
  head[0] = 0;  // 预留
  head[1] = 0;  // 入口
}
```

---

## 四、代码生成流程

### 4.1 VectorKernel代码生成

```cpp
uint8_t *VectorKernel::DoCodeGen(uint64_t core_limit, uint8_t *code_ptr, 
                                  uint64_t code_reserve) {
  uint64_t *pc = reinterpret_cast<uint64_t *>(code_ptr);
  
  // 1. 分配静态xbuf
  static_xbuf_ = code_reserve;
  for (size_t i = 0; i < static_ops_.size(); ++i) {
    auto op = static_ops_[i];
    if (i < load_num_) {
      op->xbuf_ = static_xbuf_;
      if (op->obj_id_ == kMultiLoad) {
        static_xbuf_ += xbuf_size_ + xbuf_size_;
      }
    } else {
      op->lhs_->xbuf_ = static_xbuf_;
    }
    static_xbuf_ += tile_size_ * ITEM_SIZE[op->type_id_];
  }
  
  // 2. 发射指令
  for (auto op : objects_) {
    if (op->flags_ & OBJ_FLAG_DEAD) continue;
    
    op->tail_insn_ = op->insn_ = pc;
    
    switch (op->CgTmpl()) {
      case kGenSimd0:  // 无输入
        pc += op->Emit(*this);
        break;
        
      case kGenSimd1:  // 单输入
        if (op->xbuf_ == 0) AllocOutXBuf(op);
        if (op->flags_ & OBJ_FLAG_FREE_LHS) {
          free_xbuf_.Push(op->lhs_->xbuf_, op);
        }
        pc += op->Emit(*this);
        SimdSync(op->lhs_, op);
        break;
        
      case kGenSimd2:  // 双输入
        if (op->xbuf_ == 0) AllocOutXBuf(op);
        if (op->flags_ & OBJ_FLAG_FREE_LHS) {
          free_xbuf_.Push(op->lhs_->xbuf_, op);
        }
        if (op->flags_ & OBJ_FLAG_FREE_RHS) {
          free_xbuf_.Push(op->rhs_->xbuf_, op);
        }
        pc += op->Emit(*this);
        SimdSync(op->lhs_, op);
        SimdSync(op->rhs_, op);
        break;
        
      case kGenFlex:  // Reduce等
        pc += GenFlexOpCommon(static_cast<FlexOp *>(op));
        SimdSync(op->lhs_, op);
        break;
    }
  }
  
  return reinterpret_cast<uint8_t *>(pc);
}
```

### 4.2 Load指令生成

```cpp
uint64_t NDLoad::Emit(VectorKernel &kernel) {
  vLoad op;
  op.xn = xbuf_;
  op.iter_size = kernel.LeadAlign();
  op.body_iter = tile_num_ / kernel.tile_size_;
  op.tile_stride = kernel.tile_size_;
  op.pad_size = 0;
  op.tail_iter = tile_num_ % kernel.tile_size_;
  op.round_rank = round_rank_;
  op.from = addr_.gm;
  
  return vLoad::Encode(insn_, V_LOAD, op);
}
```

### 4.3 Binary指令生成

```cpp
uint64_t NDBinary::Emit(VectorKernel &kernel) {
  auto insn_id = binary_id_list[op_type_].ids[type_id_];
  
  vBinary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = rhs_->xbuf_;
  
  return vBinary::Encode(insn_, insn_id, op);
}
```

### 4.4 Reduce指令生成

```cpp
uint64_t ReduceOp::Emit(VectorKernel &kernel) {
  auto insn_id_x = reduce_x_list[type_id_][red_type_];
  auto insn_id_y = reduce_y_list[type_id_][red_type_];
  
  // 发射X方向Reduce
  vReduce op_x;
  op_x.xd = xbuf_;
  op_x.xn = lhs_->xbuf_;
  op_x.iter_num = reduce_iter_;
  op_x.simd_width = simd_width_;
  op_x.iter_tail = iter_tail_;
  
  uint64_t len = vReduce::Encode(insn_, insn_id_x, op_x);
  
  // 发射Y方向Reduce（如果需要）
  if (need_y_reduce_) {
    vReduce op_y;
    op_y.xd = xbuf_;
    op_y.xn = xbuf_;
    // ...
    len += vReduce::Encode(insn_ + len, insn_id_y, op_y);
  }
  
  return len;
}
```

### 4.5 Cube指令生成

```cpp
uint64_t CubeOp::Emit(vCubeOp *cube_code, Tuner *tuner) {
  // 计算分块参数
  tuner->Tune(this);
  
  cube_code->flags = 0;
  if (bias_) cube_code->flags |= V_CUBE_FLAG_WITH_BIAS;
  if (trans_a_) cube_code->flags |= V_CUBE_FLAG_TRANS_A;
  if (trans_b_) cube_code->flags |= V_CUBE_FLAG_TRANS_B;
  
  cube_code->block_dim = block_dim_;
  cube_code->core_loop = core_loop_;
  
  // 设置地址（稍后重定位）
  cube_code->gm_a = 0;
  cube_code->gm_b = 0;
  cube_code->gm_c = 0;
  cube_code->gm_bias = 0;
  
  return sizeof(vCubeOp);
}
```

---

## 五、地址重定位

### 5.1 RelocAddr机制

```cpp
struct RelocAddr {
  union {
    void *gm;           // 全局内存地址
    uint64_t ws;        // Workspace偏移
    const RelocAddr *op;// 指向其他地址
    uint64_t data;      // 数据值
  };
  uint64_t *reloc_;     // 重定位指针
  
  void Reloc(void *dst) {
    if (reloc_) {
      *reloc_ = reinterpret_cast<uint64_t>(dst);
    }
  }
  
  void Update(uint64_t *insn) { reloc_ = insn; }
  void Update(const RelocAddr &share) { reloc_ = share.reloc_; }
};
```

### 5.2 重定位流程

```cpp
void Code::RelocBinds(void *workspace) {
  // 重定位Workspace地址
  for (auto ws : bind_wss_) {
    ws->Reloc(static_cast<uint8_t *>(workspace) + ws->ws);
  }
  
  // 重定位操作数地址
  for (auto op : bind_ops_) {
    op->Reloc(op->gm);
  }
}

// 在CodeGen时绑定
void Code::BindOpFast(RelocAddr &addr, RelocAddr &src) {
  addr.Update(src);
  bind_ops_.push_back(&addr);
}
```

### 5.3 动态Shape重定位

```cpp
struct RelocEntry {
  NDObject *io;    // 输入/输出对象
  void *addr;      // 实际地址
};

void Kernel::CodeGen(const RelocEntry *relocs, size_t reloc_size, 
                     WsAllocator *ws_alloc) {
  // 应用重定位
  auto reloc = relocs;
  for (size_t i = 0; i < reloc_size; ++i, ++reloc) {
    static_cast<NDAccess *>(reloc->io)->addr_.Reloc(reloc->addr);
  }
  
  // 分配Workspace
  if (ws_alloc) {
    pre_ws_mem_ = ws_alloc->Alloc(pre_ws_size_);
  }
}
```

---

## 六、反汇编系统

### 6.1 反汇编接口

```cpp
void Code::DisAssemble(std::ostringstream &oss) {
  uint64_t *insn = reinterpret_cast<uint64_t *>(data_ + HeadSize());
  uint64_t *end = reinterpret_cast<uint64_t *>(data_ + data_size_);
  
  while (insn < end) {
    auto id = static_cast<vSimdInsnID>(*insn & 0xFF);
    
    switch (id) {
      case V_LOAD:
        DumpLoad(insn, oss);
        insn += 3;
        break;
      case V_STORE:
        DumpStore(insn, oss);
        insn += 3;
        break;
      case V_ADD:
      case V_SUB:
      case V_MUL:
        DumpBinary(insn, id, oss);
        insn += 1;
        break;
      // ... 更多指令
    }
    
    oss << "\n";
  }
}
```

### 6.2 指令Dump实现

```cpp
void DumpLoad(uint64_t *insn, std::ostringstream &oss) {
  vLoad op;
  vLoad::Decode(insn, *insn, op);
  
  oss << "load.u8." << op.iter_size << "x" << op.body_iter;
  oss << " " << reinterpret_cast<void *>(op.xn) << ", " 
      << reinterpret_cast<void *>(op.from);
  oss << " // ";
  DumpVal("tile_stride", op.tile_stride, oss);
  oss << ", ";
  DumpVal("pad_size", op.pad_size, oss);
  oss << ", ";
  DumpVal("iter_tail", op.tail_iter, oss);
}

void DumpBinary(uint64_t *insn, vSimdInsnID id, std::ostringstream &oss) {
  vBinary op;
  vBinary::Decode(insn, *insn, op);
  
  static const char *names[] = {
    "copy", "copy_cube_tile", "nop", "broadcast_y", "broadcast_s",
    "sqrt", "abs", "log", "exp", "round", "floor", "ceil", "trunc",
    "adds", "muls", "divs", "sdiv", "cmps",
    "add", "sub", "mul", "div", "min", "max", "cmp",
    ...
  };
  
  oss << names[id] << " " << reinterpret_cast<void *>(op.xd) << ", "
      << reinterpret_cast<void *>(op.xn) << ", "
      << reinterpret_cast<void *>(op.xm);
}
```

---

## 七、Workspace管理

### 7.1 Workspace分配

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
  
  std::vector<void *> dev_mem_;
};
```

### 7.2 Workspace布局

```
┌─────────────────────────────────────────────────────────┐
│                    Workspace布局                         │
├─────────────────────────────────────────────────────────┤
│  [0, xbuf_size)        : 输入缓冲区                      │
│  [xbuf_size, 2*xbuf)   : 输出缓冲区                      │
│  [2*xbuf, ...)         : 中间结果缓冲区                  │
│  [...]                 : 通信缓冲区                      │
└─────────────────────────────────────────────────────────┘
```

---

## 八、代码执行

### 8.1 Launch流程

```cpp
int Code::Launch(void *workspace, void *stream) {
  // 1. 重定位绑定
  RelocBinds(workspace);
  
  // 2. 获取入口函数
  auto entry = reinterpret_cast<uint64_t *>(data_);
  auto entry_data = entry[1];
  
  // 3. 根据入口类型执行
  auto entry_type = entry_data & 0xC000000000000000UL;
  
  switch (entry_type) {
    case V_ENTRY_TYPE_V:
      return LaunchVector(workspace, stream);
    case V_ENTRY_TYPE_C:
      return LaunchCube(workspace, stream);
    case V_ENTRY_TYPE_VE:
      return LaunchReduce(workspace, stream);
  }
  
  return 0;
}
```

### 8.2 Vector执行

```cpp
int LaunchVector(void *workspace, void *stream) {
  // 解析入口参数
  auto entry = entry_data & ~0xC000000000000000UL;
  auto block_tile = entry >> V_ENTRY_V_TILE_BODY_OFFSET;
  auto block_tail = (entry >> V_ENTRY_V_TILE_TAIL_OFFSET) & 0xFFFF;
  
  // 设置Kernel参数
  vKernelArgs args;
  args.code = data_ + HeadSize();
  args.workspace = workspace;
  args.block_tile = block_tile;
  args.block_tail = block_tail;
  
  // 下发到Device
  return aclrtLaunchKernel(vmain_vector, &args, block_dim_, stream);
}
```

---

## 九、总结

本文详细解析了DVM的字节码生成机制：

1. **指令集架构**：访存、SIMD、访问模式三大类指令
2. **指令编码**：64位定长编码，支持多种操作数格式
3. **Code类**：字节码容器，支持分配、发射、执行
4. **代码生成**：从计算图到字节码的完整流程
5. **地址重定位**：支持动态Shape和地址绑定
6. **反汇编**：调试友好的指令Dump
7. **Workspace**：高效的内存管理

字节码生成是连接Host侧编译和Device侧执行的桥梁，理解它对于深入掌握DVM至关重要。

---

## 参考资料

- [isa.h](../src/isa.h)
- [code.h](../src/code.h)
- [code.cc](../src/code.cc)
- [kernel.cc](../src/kernel.cc)
