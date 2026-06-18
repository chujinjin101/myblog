---
title: DVM虚拟机：AIV核执行引擎
date: 2026-05-19 16:33:04
tags:
---

# DVM虚拟机：AIV核执行引擎

## 前言

DVM虚拟机是运行在Ascend NPU上的字节码解释执行引擎。它接收Host侧生成的字节码指令，在Device侧进行解释执行。本文将深入解析AIV（AI Vector）核虚拟机的实现原理，揭示DVM如何在微秒级完成算子编译和执行。

---

## 一、虚拟机概述

### 1.1 执行架构

```
┌─────────────────────────────────────────────────────────┐
│                    Host侧                                │
│  用户构图 → Normalize → CodeGen → 字节码                │
└─────────────────────────────────────────────────────────┘
                        │ Launch
                        ↓
┌─────────────────────────────────────────────────────────┐
│                    Device侧                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │              AIV核虚拟机                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │   │
│  │  │  Core0  │  │  Core1  │  │  Core2  │ ...     │   │
│  │  │ vmain() │  │ vmain() │  │ vmain() │         │   │
│  │  └─────────┘  └─────────┘  └─────────┘         │   │
│  └─────────────────────────────────────────────────┘   │
│                        │                                │
│                        ↓                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │              内存层次                            │   │
│  │  GM (Global Memory) ←→ UB (Unified Buffer)      │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 1.2 核心特点

| 特点 | 说明 |
|-----|------|
| Tile粒度执行 | 将大张量切分为Tile逐块处理 |
| 多核并行 | 多个AIV Core并行执行 |
| Post-Execution | 隐藏解释执行开销 |
| UB内存复用 | 高效的内存管理策略 |

---

## 二、虚拟机入口

### 2.1 入口函数

```cpp
// vm_aiv.cce
extern "C" __aicore__ void vmain_vector(
    uint8_t *code,        // 字节码指针
    void *workspace,      // Workspace地址
    uint64_t block_tile,  // 每个Block处理的Tile数
    uint64_t block_tail   // 尾部Block数量
);

extern "C" __aicore__ void vmain_reduce(
    uint8_t *code,
    void *workspace,
    uint64_t visit_id,
    uint64_t visit_offset
);

extern "C" __aicore__ void vmain_cube(
    uint8_t *code,
    void *workspace
);
```

### 2.2 入口编码格式

```cpp
// 入口类型
#define V_ENTRY_TYPE_V    0x0000000000000000UL  // Vector入口
#define V_ENTRY_TYPE_C    0x4000000000000000UL  // Cube入口
#define V_ENTRY_TYPE_VE   0x8000000000000000UL  // Reduce入口

// Vector入口参数
#define V_ENTRY_V_TILE_BODY_OFFSET   0
#define V_ENTRY_V_TILE_TAIL_OFFSET   16

// Reduce入口参数
#define V_ENTRY_VE_VISIT_ID_OFFSET     0
#define V_ENTRY_VE_VISIT_OFFSET_OFFSET 16
```

---

## 三、Vector虚拟机实现

### 3.1 主执行循环

```cpp
__aicore__ void vmain_vector(
    uint8_t *code,
    void *workspace,
    uint64_t block_tile,
    uint64_t block_tail
) {
  // 获取Block ID
  auto block_id = GetBlockIdx();
  
  // 计算Tile范围
  uint64_t tile_start = block_id * block_tile;
  uint64_t tile_end = tile_start + block_tile;
  
  // 处理尾部
  if (block_id < block_tail) {
    tile_end--;
  }
  
  // 执行每个Tile
  for (uint64_t tile = tile_start; tile < tile_end; ++tile) {
    ExecuteTile(code, workspace, tile);
  }
}

__aicore__ void ExecuteTile(uint8_t *code, void *workspace, uint64_t tile) {
  uint64_t *pc = reinterpret_cast<uint64_t *>(code);
  
  while (true) {
    auto insn = *pc;
    auto op = static_cast<vSimdInsnID>(insn & 0xFF);
    
    switch (op) {
      case V_LOAD:
        ExecLoad(pc, workspace, tile);
        pc += 3;
        break;
        
      case V_STORE:
        ExecStore(pc, workspace, tile);
        pc += 3;
        break;
        
      case V_ADD:
      case V_SUB:
      case V_MUL:
      case V_DIV:
        ExecBinary(pc, op);
        pc += 1;
        break;
        
      case V_SQRT:
      case V_EXP:
      case V_LOG:
        ExecUnary(pc, op);
        pc += 1;
        break;
        
      case V_RSUM_X:
      case V_RMAX_X:
        ExecReduceX(pc);
        pc += 1;
        break;
        
      case V_NOP:
        return;  // 结束
        
      default:
        pc += 1;
        break;
    }
  }
}
```

### 3.2 Load指令执行

```cpp
__aicore__ void ExecLoad(uint64_t *insn, void *workspace, uint64_t tile) {
  vLoad op;
  vLoad::Decode(insn, op);
  
  // 计算UB地址
  auto ub_addr = reinterpret_cast<uint8_t *>(workspace) + op.xn;
  
  // 计算GM地址
  auto gm_addr = reinterpret_cast<uint8_t *>(op.from) + tile * op.tile_stride;
  
  // 执行加载
  if (op.tail_iter && tile >= op.body_iter) {
    // 尾部处理
    Copy(ub_addr, gm_addr, op.tail_iter * op.iter_size);
  } else {
    // 主体处理
    Copy(ub_addr, gm_addr, op.iter_size);
  }
}
```

### 3.3 Store指令执行

```cpp
__aicore__ void ExecStore(uint64_t *insn, void *workspace, uint64_t tile) {
  vStore op;
  vStore::Decode(insn, op);
  
  // 计算UB地址
  auto ub_addr = reinterpret_cast<uint8_t *>(workspace) + op.xn;
  
  // 计算GM地址
  auto gm_addr = reinterpret_cast<uint8_t *>(op.to) + tile * op.tile_stride;
  
  // 执行存储
  if (op.tail_iter && tile >= op.body_iter) {
    Copy(gm_addr, ub_addr, op.tail_iter * op.iter_size);
  } else {
    Copy(gm_addr, ub_addr, op.iter_size);
  }
}
```

### 3.4 Binary指令执行

```cpp
__aicore__ void ExecBinary(uint64_t *insn, vSimdInsnID op) {
  vBinary binary;
  vBinary::Decode(insn, binary);
  
  auto xd = reinterpret_cast<half *>(binary.xd);
  auto xn = reinterpret_cast<half *>(binary.xn);
  auto xm = reinterpret_cast<half *>(binary.xm);
  
  switch (op) {
    case V_ADD:
      for (int i = 0; i < SIMD_WIDTH; ++i) {
        xd[i] = xn[i] + xm[i];
      }
      break;
      
    case V_SUB:
      for (int i = 0; i < SIMD_WIDTH; ++i) {
        xd[i] = xn[i] - xm[i];
      }
      break;
      
    case V_MUL:
      for (int i = 0; i < SIMD_WIDTH; ++i) {
        xd[i] = xn[i] * xm[i];
      }
      break;
      
    case V_DIV:
      for (int i = 0; i < SIMD_WIDTH; ++i) {
        xd[i] = xn[i] / xm[i];
      }
      break;
      
    // ... 更多操作
  }
}
```

### 3.5 Reduce指令执行

```cpp
__aicore__ void ExecReduceX(uint64_t *insn) {
  vReduce op;
  vReduce::Decode(insn, op);
  
  auto xd = reinterpret_cast<float *>(op.xd);
  auto xn = reinterpret_cast<float *>(op.xn);
  
  // 初始化累加器
  float acc = 0.0f;
  
  // X方向归约
  for (uint64_t i = 0; i < op.iter_num; ++i) {
    for (uint64_t j = 0; j < op.simd_width; ++j) {
      acc += xn[i * op.simd_width + j];
    }
  }
  
  // 处理尾部
  for (uint64_t j = 0; j < op.iter_tail; ++j) {
    acc += xn[op.iter_num * op.simd_width + j];
  }
  
  *xd = acc;
}
```

---

## 四、Reduce虚拟机实现

### 4.1 Reduce访问模式

```cpp
// 访问模式定义
enum VisitMode {
  V_VISIT_RED_1 = 0,  // 单Reduce维度
  V_VISIT_RED_2,      // 双Reduce维度
  V_VISIT_RED_3,      // 三Reduce维度
  V_VISIT_RED_4,      // 四Reduce维度
};

// 访问函数表
extern "C" __aicore__ VisitFunc g_visit_funcs[];

typedef void (*VisitFunc)(uint64_t tile, void *workspace, uint8_t *code);
```

### 4.2 Reduce入口

```cpp
__aicore__ void vmain_reduce(
    uint8_t *code,
    void *workspace,
    uint64_t visit_id,
    uint64_t visit_offset
) {
  // 获取Block ID
  auto block_id = GetBlockIdx();
  
  // 计算Tile
  uint64_t tile = block_id + visit_offset;
  
  // 调用访问函数
  g_visit_funcs[visit_id](tile, workspace, code);
}
```

### 4.3 访问函数实现

```cpp
// 单维度Reduce访问
__aicore__ void VisitRed1(uint64_t tile, void *workspace, uint8_t *code) {
  // 计算Reduce索引
  uint64_t red_idx = tile % reduce_dim_;
  
  // 执行Reduce操作
  ExecuteReduceTile(code, workspace, red_idx);
}

// 双维度Reduce访问
__aicore__ void VisitRed2(uint64_t tile, void *workspace, uint8_t *code) {
  uint64_t red_idx0 = tile % reduce_dim0_;
  uint64_t red_idx1 = (tile / reduce_dim0_) % reduce_dim1_;
  
  ExecuteReduceTile2D(code, workspace, red_idx0, red_idx1);
}
```

---

## 五、Cube虚拟机实现

### 5.1 Cube入口

```cpp
__aicore__ void vmain_cube(uint8_t *code, void *workspace) {
  auto block_id = GetBlockIdx();
  
  // 解析Cube指令
  vCubeOp *cube = reinterpret_cast<vCubeOp *>(code);
  
  if (block_id < cube->block_dim) {
    ExecuteCube(cube, block_id);
  }
}
```

### 5.2 Cube执行

```cpp
__aicore__ void ExecuteCube(vCubeOp *cube, uint64_t block_id) {
  // 获取矩阵参数
  auto gm_a = reinterpret_cast<half *>(cube->gm_a);
  auto gm_b = reinterpret_cast<half *>(cube->gm_b);
  auto gm_c = reinterpret_cast<float *>(cube->gm_c);
  
  // 分块计算
  uint64_t m_start = block_id * m_per_core;
  uint64_t m_end = std::min(m_start + m_per_core, total_m);
  
  for (uint64_t m = m_start; m < m_end; ++m) {
    for (uint64_t n = 0; n < total_n; ++n) {
      float sum = 0.0f;
      
      for (uint64_t k = 0; k < total_k; ++k) {
        half a_val = cube->flags & V_CUBE_FLAG_TRANS_A ? gm_a[k * lda + m] : gm_a[m * lda + k];
        half b_val = cube->flags & V_CUBE_FLAG_TRANS_B ? gm_b[n * ldb + k] : gm_b[k * ldb + n];
        sum += static_cast<float>(a_val) * static_cast<float>(b_val);
      }
      
      // 添加Bias
      if (cube->flags & V_CUBE_FLAG_WITH_BIAS) {
        auto gm_bias = reinterpret_cast<float *>(cube->gm_bias);
        sum += gm_bias[n];
      }
      
      gm_c[m * ldc + n] = sum;
    }
  }
}
```

---

## 六、内存管理

### 6.1 UB内存布局

```
┌─────────────────────────────────────────────────────────┐
│                    UB内存布局                            │
├─────────────────────────────────────────────────────────┤
│  [0, tile_size)           : 输入缓冲区0                 │
│  [tile_size, 2*tile_size) : 输入缓冲区1                 │
│  [2*tile_size, 3*tile_size): 输出缓冲区                 │
│  [...]                    : 中间结果                    │
└─────────────────────────────────────────────────────────┘
```

### 6.2 UB内存复用

```cpp
// Host侧分配xbuf
void AllocOutXBuf(NDObject *op) {
  // 尝试复用输入xbuf
  if (op->flags_ & F_LR && op->lhs_->flags_ & OBJ_FLAG_FREE_LHS) {
    op->xbuf_ = op->lhs_->xbuf_;
    return;
  }
  
  if (op->flags_ & F_RR && op->rhs_->flags_ & OBJ_FLAG_FREE_RHS) {
    op->xbuf_ = op->rhs_->xbuf_;
    return;
  }
  
  // 从空闲列表分配
  op->xbuf_ = free_xbuf_.Pop(op);
}
```

### 6.3 GM访问优化

```cpp
// 双缓冲加载
__aicore__ void PingPongLoad(uint8_t *ub_ping, uint8_t *ub_pong, 
                              uint8_t *gm, uint64_t tile_size) {
  // 预加载第一块
  CopyAsync(ub_ping, gm, tile_size);
  
  for (uint64_t i = 0; i < total_tiles - 1; ++i) {
    // 等待前一块完成
    WaitCopy();
    
    // 异步加载下一块
    CopyAsync(ub_pong, gm + (i + 1) * tile_size, tile_size);
    
    // 处理当前块
    ProcessTile(ub_ping);
    
    // 交换缓冲区
    std::swap(ub_ping, ub_pong);
  }
  
  // 处理最后一块
  WaitCopy();
  ProcessTile(ub_ping);
}
```

---

## 七、Post-Execution机制

### 7.1 设计思想

Post-Execution通过流水线隐藏解释执行开销：

```
传统执行：
Tile0解释 → Tile0执行 → Tile1解释 → Tile1执行 → ...

Post-Execution：
Tile0解释 → Tile1解释 → Tile2解释 → ...
            Tile0执行 → Tile1执行 → Tile2执行 → ...
```

### 7.2 实现代码

```cpp
__aicore__ void ExecuteWithPostExec(uint8_t *code, void *workspace, uint64_t num_tiles) {
  uint64_t *pc = reinterpret_cast<uint64_t *>(code);
  uint64_t tile = 0;
  
  // 预解释前几个Tile
  for (int i = 0; i < POST_EXEC_DEPTH && tile < num_tiles; ++i, ++tile) {
    InterpretTile(pc, workspace, tile);
  }
  
  // 流水线执行
  while (tile < num_tiles) {
    // 等待前一个Tile执行完成
    WaitExecution();
    
    // 解释下一个Tile
    InterpretTile(pc, workspace, tile);
    
    // 启动当前Tile执行
    StartExecution();
    
    ++tile;
  }
  
  // 等待所有执行完成
  WaitAllExecutions();
}
```

---

## 八、SIMD指令映射

### 8.1 Ascend SIMD指令

DVM虚拟机将字节码指令映射到Ascend SIMD指令：

```cpp
// Ascend SIMD指令示例
__aicore__ void vadd(half *dst, half *src0, half *src1, uint64_t num) {
  for (uint64_t i = 0; i < num; i += SIMD_WIDTH) {
    // 使用Ascend向量指令
    __asm__ volatile (
      "vadd.half %0, %1, %2\n"
      : "=v"(dst[i])
      : "v"(src0[i]), "v"(src1[i])
    );
  }
}

__aicore__ void vmul(half *dst, half *src0, half *src1, uint64_t num) {
  for (uint64_t i = 0; i < num; i += SIMD_WIDTH) {
    __asm__ volatile (
      "vmul.half %0, %1, %2\n"
      : "=v"(dst[i])
      : "v"(src0[i]), "v"(src1[i])
    );
  }
}
```

### 8.2 指令映射表

```cpp
// 指令映射表
struct SimdEntry {
  vSimdInsnID id;
  SimdFunc func;
};

SimdEntry simd_table[] = {
  {V_ADD, ExecAdd},
  {V_SUB, ExecSub},
  {V_MUL, ExecMul},
  {V_DIV, ExecDiv},
  {V_SQRT, ExecSqrt},
  {V_EXP, ExecExp},
  {V_LOG, ExecLog},
  // ...
};

__aicore__ void ExecAdd(half *dst, half *src0, half *src1, uint64_t num) {
  vadd(dst, src0, src1, num);
}
```

---

## 九、性能优化

### 9.1 多核并行

```cpp
// Host侧设置Block维度
void SetBlockDim(uint64_t core_limit) {
  // 根据Tile数量和核心数计算
  uint64_t tiles_per_core = CeilDiv(tile_num_, core_limit);
  block_dim_ = CeilDiv(tile_num_, tiles_per_core);
  
  // 限制最大核心数
  block_dim_ = std::min(block_dim_, core_limit);
}
```

### 9.2 内存对齐

```cpp
// 计算对齐后的Tile大小
uint64_t AlignTileSize(uint64_t raw_size, DataType type) {
  // 32字节对齐
  uint64_t align = 32 / ITEM_SIZE[type];
  return (raw_size + align - 1) & ~(align - 1);
}
```

### 9.3 Bank冲突避免

```cpp
// UB Bank分配
uint64_t AllocXBufBank(uint64_t size) {
  // 轮询分配到不同Bank
  static uint64_t bank = 0;
  uint64_t offset = bank * BANK_SIZE;
  bank = (bank + 1) % NUM_BANKS;
  return offset;
}
```

---

## 十、总结

本文详细解析了DVM AIV核虚拟机的实现：

1. **执行架构**：Host侧编译 → Device侧解释执行
2. **入口函数**：Vector、Reduce、Cube三种入口
3. **指令执行**：Load、Store、Binary、Unary、Reduce等
4. **内存管理**：UB布局、内存复用、双缓冲
5. **Post-Execution**：流水线隐藏解释开销
6. **SIMD映射**：字节码到Ascend指令的映射
7. **性能优化**：多核并行、内存对齐、Bank冲突避免

DVM虚拟机是算子执行的核心，其高效的解释执行机制是DVM实现微秒级编译的关键。

---

## 参考资料

- [vm_aiv.h](../src/vm_aiv.h)
- [vm_aiv.cce](../src/vm_aiv.cce)
- [isa.h](../src/isa.h)
- [code.h](../src/code.h)
