---
title: DVM性能优化：Tiling策略与内存管理
date: 2026-05-19 16:33:04
tags:
---

# DVM性能优化：Tiling策略与内存管理

## 前言

Tiling（分块）策略和内存管理是DVM性能优化的核心。合理的Tiling策略可以最大化数据局部性，高效的内存管理可以降低内存占用和访问延迟。本文将深入解析DVM的Tiling策略和内存管理机制。

---

## 一、Tiling基础

### 1.1 为什么需要Tiling

在深度学习算子中，数据规模往往远大于处理器的缓存容量。Tiling将大数据切分为小块，使每个小块能够完全放入缓存中处理：

```
原始数据：[1024 x 1024] = 1M元素

Tiling后：
┌────────┬────────┬────────┬────────┐
│ Tile0  │ Tile1  │ Tile2  │ Tile3  │
│ [128]  │ [128]  │ [128]  │ [128]  │
├────────┼────────┼────────┼────────┤
│ Tile4  │ Tile5  │ Tile6  │ Tile7  │
│ [128]  │ [128]  │ [128]  │ [128]  │
├────────┼────────┼────────┼────────┤
│  ...   │  ...   │  ...   │  ...   │
└────────┴────────┴────────┴────────┘

每个Tile = [128 x 128] = 16K元素
```

### 1.2 Tiling目标

| 目标 | 说明 |
|-----|------|
| 数据局部性 | 最大化UB命中率，减少GM访问 |
| 并行度 | 合理分配Tile到多核 |
| 内存效率 | 最小化UB占用峰值 |
| 负载均衡 | 各核心工作量均衡 |

---

## 二、DVM Tiling策略

### 2.1 Tile参数

```cpp
struct TileParam {
  uint64_t tile_num;    // Tile总数
  uint64_t tile_size;   // 每个Tile的元素数
  uint64_t xbuf_size;   // UB缓冲区大小
  uint64_t lead_align;  // 前导对齐
  int max_type;         // 最大数据类型
  int min_type;         // 最小数据类型
};
```

### 2.2 Tile计算流程

```cpp
void VectorKernel::PrepareTiling() {
  // 1. 构建计算域
  BuildDomain();
  
  // 2. 计算Tile数量
  auto &dims = dom_->nd_.dims();
  tile_num_ = 1;
  for (auto dim : dims) {
    tile_num_ *= dim;
  }
  
  // 3. 计算Tile大小
  CalculateTileSize();
  
  // 4. 计算UB需求
  CalculateUBSize();
}
```

### 2.3 Tile大小计算

```cpp
void VectorKernel::CalculateTileSize() {
  // 获取UB容量
  uint64_t ub_capacity = GetUBCapacity();
  
  // 计算最大Tile大小
  uint64_t max_tile = ub_capacity / (ITEM_SIZE[max_type_] * max_live_vars_);
  
  // 考虑对齐
  uint64_t align = 32 / ITEM_SIZE[max_type_];
  tile_size_ = (max_tile / align) * align;
  
  // 限制Tile数量上限
  uint64_t max_tiles = GetMaxTiles();
  if (tile_num_ > max_tiles) {
    tile_size_ = CeilDiv(tile_num_, max_tiles);
    tile_num_ = max_tiles;
  }
}
```

### 2.4 维度Tiling

```cpp
struct DimTile {
  int dim;              // 维度索引
  int64_t tile_size;    // Tile大小
  int64_t num_tiles;    // Tile数量
  int64_t tail_size;    // 尾部大小
};

void SetTile(int start, int end, int64_t num, int64_t factor) {
  tiles_.clear();
  
  for (int dim = start; dim < end; ++dim) {
    DimTile tile;
    tile.dim = dim;
    tile.tile_size = factor > 0 ? factor : dom_->dims()[dim];
    tile.num_tiles = CeilDiv(dom_->dims()[dim], tile.tile_size);
    tile.tail_size = dom_->dims()[dim] % tile.tile_size;
    
    tiles_.push_back(tile);
  }
  
  // 更新Tile总数
  tile_num_ = 1;
  for (auto &tile : tiles_) {
    tile_num_ *= tile.num_tiles;
  }
}
```

---

## 三、UB内存管理

### 3.1 UB布局规划

```cpp
void VectorKernel::PlanUBLayout() {
  // 静态分配
  static_xbuf_ = 0;
  
  // 1. 分配Load缓冲区
  for (auto op : objects_) {
    if (op->IsLoad()) {
      op->xbuf_ = static_xbuf_;
      static_xbuf_ += tile_size_ * ITEM_SIZE[op->type_id_];
    }
  }
  
  // 2. 分配中间结果缓冲区
  for (auto op : objects_) {
    if (op->IsSimd() && !op->IsStore()) {
      if (op->xbuf_ == 0) {
        // 尝试复用
        if (!TryReuseXBuf(op)) {
          op->xbuf_ = static_xbuf_;
          static_xbuf_ += tile_size_ * ITEM_SIZE[op->type_id_];
        }
      }
    }
  }
  
  // 3. 分配Store缓冲区
  for (auto op : objects_) {
    if (op->IsStore()) {
      op->xbuf_ = op->lhs_->xbuf_;
    }
  }
  
  xbuf_size_ = static_xbuf_;
}
```

### 3.2 内存复用策略

```cpp
bool TryReuseXBuf(NDObject *op) {
  // 策略1：Inplace复用
  if (op->flags_ & F_IP) {
    if (op->lhs_ && op->lhs_->flags_ & OBJ_FLAG_FREE_LHS) {
      op->xbuf_ = op->lhs_->xbuf_;
      return true;
    }
    if (op->rhs_ && op->rhs_->flags_ & OBJ_FLAG_FREE_RHS) {
      op->xbuf_ = op->rhs_->xbuf_;
      return true;
    }
  }
  
  // 策略2：从空闲列表分配
  auto xbuf = free_xbuf_.Pop(op);
  if (xbuf != 0) {
    op->xbuf_ = xbuf;
    return true;
  }
  
  return false;
}
```

### 3.3 空闲列表管理

```cpp
class FreeXBufList {
 public:
  uint64_t Pop(NDObject *user) {
    if (list_.empty()) {
      return 0;
    }
    
    // 找到最早可用的xbuf
    for (auto it = list_.begin(); it != list_.end(); ++it) {
      if (it->first <= user->index_) {
        auto xbuf = it->second;
        list_.erase(it);
        return xbuf;
      }
    }
    
    return 0;
  }
  
  void Push(uint64_t xbuf, NDObject *owner) {
    // 找到所有用户中最晚的索引
    uint64_t last_use = owner->index_;
    for (auto user : GetUsers(owner)) {
      last_use = std::max(last_use, user->index_);
    }
    
    list_.emplace_back(last_use, xbuf);
  }
  
 private:
  std::list<std::pair<uint64_t, uint64_t>> list_;  // (last_use, xbuf)
};
```

---

## 四、活跃变量分析

### 4.1 活跃变量定义

```
活跃变量：在某个程序点之后还会被使用的变量

程序点：Load -> Binary -> Binary -> Store
        │        │         │         │
活跃变量：  {L0}    {L0,L1}  {L0,L1}   {S0}
```

### 4.2 活跃变量计算

```cpp
size_t CalculatePeakLiveness(VectorKernel &kernel) {
  size_t peak = 0;
  size_t current = 0;
  std::unordered_map<NDObject *, int> ref_count;
  
  // 计算引用计数
  for (auto op : kernel.objects_) {
    op->ForInput([&ref_count](NDObject *input) {
      ref_count[input]++;
    });
  }
  
  // 遍历计算峰值
  for (auto op : kernel.objects_) {
    // 新产生变量
    if (!op->IsLoad() && !op->IsStore()) {
      current++;
    }
    
    peak = std::max(peak, current);
    
    // 释放不再使用的变量
    op->ForInput([&](NDObject *input) {
      if (--ref_count[input] == 0) {
        current--;
      }
    });
  }
  
  return peak;
}
```

### 4.3 峰值优化

```cpp
void OptimizePeakLiveness(BasicBlock &bb) {
  // 使用启发式重排
  auto new_order = ReorderForMinPeak(bb);
  
  // 检查是否改善
  auto old_peak = CalculatePeakLiveness(bb);
  bb.ApplyOrder(new_order);
  auto new_peak = CalculatePeakLiveness(bb);
  
  // 如果没有改善，恢复原顺序
  if (new_peak >= old_peak) {
    bb.RestoreOrder();
  }
}
```

---

## 五、多核并行策略

### 5.1 Block维度计算

```cpp
uint32_t VectorKernel::CompactBlockDim(uint64_t core_limit) {
  // 计算每个核心处理的Tile数
  uint64_t tiles_per_core = CeilDiv(tile_num_, core_limit);
  
  // 计算实际需要的核心数
  uint32_t block_dim = CeilDiv<uint32_t>(tile_num_, tiles_per_core);
  
  // 限制最大核心数
  return std::min(block_dim, static_cast<uint32_t>(core_limit));
}
```

### 5.2 负载均衡

```cpp
void BalanceWorkload(uint64_t tile_num, uint64_t block_dim,
                     uint64_t &body_tiles, uint64_t &tail_blocks) {
  // 计算基本分配
  body_tiles = CeilDiv(tile_num, block_dim);
  
  // 计算尾部块数
  uint64_t total_with_tail = body_tiles * block_dim;
  tail_blocks = total_with_tail - tile_num;
}
```

### 5.3 并行执行示意

```
Tile总数 = 100，核心数 = 8

方案1（不均衡）：
Core0: 13 tiles
Core1: 13 tiles
Core2: 12 tiles
Core3: 12 tiles
Core4: 13 tiles
Core5: 12 tiles
Core6: 12 tiles
Core7: 13 tiles

方案2（均衡）：
body_tiles = 13
tail_blocks = 4

Core0-3: 12 tiles (body_tiles - 1)
Core4-7: 13 tiles (body_tiles)
```

---

## 六、MatMul Tiling策略

### 6.1 MatMul分块

```cpp
struct MatMulTile {
  uint64_t M_tile;    // M维度Tile大小
  uint64_t N_tile;    // N维度Tile大小
  uint64_t K_tile;    // K维度Tile大小
};

MatMulTile CalculateMatMulTile(uint64_t M, uint64_t N, uint64_t K,
                                uint64_t ub_size) {
  MatMulTile tile;
  
  // L1 Cache大小限制
  uint64_t l1_size = GetL1CacheSize();
  
  // 计算最大分块
  // A矩阵：M_tile * K_tile
  // B矩阵：K_tile * N_tile
  // C矩阵：M_tile * N_tile
  
  uint64_t elem_size = 2;  // FP16
  uint64_t max_mn = static_cast<uint64_t>(std::sqrt(l1_size / elem_size / 3));
  
  tile.M_tile = std::min(M, max_mn);
  tile.N_tile = std::min(N, max_mn);
  tile.K_tile = std::min(K, l1_size / elem_size / (tile.M_tile + tile.N_tile));
  
  return tile;
}
```

### 6.2 Cube核并行

```cpp
void CubeKernel::CalculateBlockDim() {
  // M维度分块
  uint64_t m_tiles = CeilDiv(M_, M_tile_);
  
  // N维度分块
  uint64_t n_tiles = CeilDiv(N_, N_tile_);
  
  // 总分块数
  uint64_t total_tiles = m_tiles * n_tiles;
  
  // 核心数
  block_dim_ = std::min(total_tiles, GetAvailableCores());
}
```

---

## 七、Reduce Tiling策略

### 7.1 Reduce维度处理

```cpp
void ReduceOp::CalculateTiling() {
  // Reduce维度
  auto &reduce_dims = reduce_dims_;
  
  // 非Reduce维度
  auto &keep_dims = keep_dims_;
  
  // 计算Tile
  // 非Reduce维度作为并行维度
  tile_num_ = 1;
  for (auto dim : keep_dims) {
    tile_num_ *= shape_[dim];
  }
  
  // Reduce维度作为迭代维度
  reduce_iter_ = 1;
  for (auto dim : reduce_dims) {
    reduce_iter_ *= shape_[dim];
  }
}
```

### 7.2 Reduce访问模式

```cpp
// 单维度Reduce
void VisitRed1(uint64_t tile, uint64_t reduce_dim, uint64_t reduce_size) {
  uint64_t outer_idx = tile / reduce_size;
  uint64_t inner_idx = tile % reduce_size;
  
  // 计算数据偏移
  uint64_t offset = outer_idx * reduce_size * stride_ + inner_idx;
  
  ProcessReduce(offset);
}

// 多维度Reduce
void VisitRed2(uint64_t tile, uint64_t dim0, uint64_t dim1,
               uint64_t size0, uint64_t size1) {
  uint64_t idx0 = tile % size0;
  uint64_t idx1 = (tile / size0) % size1;
  uint64_t outer = tile / (size0 * size1);
  
  uint64_t offset = CalculateOffset(outer, idx0, idx1);
  
  ProcessReduce(offset);
}
```

---

## 八、通信算子优化

### 8.1 AllReduce Tiling

```cpp
void AllReduceOp::CalculateTiling() {
  // 数据大小
  uint64_t data_size = NumElements() * ITEM_SIZE[type_];
  
  // 通信缓冲区大小限制
  uint64_t comm_buf_size = GetCommBufferSize();
  
  // 计算分块数
  if (data_size <= comm_buf_size) {
    tile_num_ = 1;
    tile_size_ = data_size;
  } else {
    tile_num_ = CeilDiv(data_size, comm_buf_size);
    tile_size_ = comm_buf_size;
  }
}
```

### 8.2 通信与计算重叠

```cpp
void ExecuteAllReduceOverlap(uint8_t *code, void *workspace) {
  for (uint64_t tile = 0; tile < tile_num_; ++tile) {
    // 1. 计算当前Tile
    ExecuteCompute(code, workspace, tile);
    
    // 2. 异步通信上一个Tile
    if (tile > 0) {
      AllReduceAsync(workspace, tile - 1);
    }
    
    // 3. 等待上上个Tile通信完成
    if (tile > 1) {
      WaitAllReduce(tile - 2);
    }
  }
  
  // 等待最后两个Tile
  WaitAllReduce(tile_num_ - 2);
  WaitAllReduce(tile_num_ - 1);
}
```

---

## 九、性能调优

### 9.1 Tuning框架

```cpp
class Tuner {
 public:
  virtual void Tune(CubeOp *cube) = 0;
  
 protected:
  void UpdateBest(const TuneResult &result);
  TuneResult best_;
};

// 在线Tuning
class OnlineTuner : public Tuner {
 public:
  void Tune(CubeOp *cube) override {
    // 尝试多种配置
    for (auto config : configs_) {
      auto result = RunWithConfig(cube, config);
      if (result.time < best_.time) {
        UpdateBest(result);
      }
    }
    
    // 应用最优配置
    ApplyBest(cube);
  }
};
```

### 9.2 配置搜索空间

```cpp
struct TuneConfig {
  uint64_t M_tile;
  uint64_t N_tile;
  uint64_t K_tile;
  uint64_t block_dim;
  uint64_t double_buffer;
};

std::vector<TuneConfig> GenerateConfigs(uint64_t M, uint64_t N, uint64_t K) {
  std::vector<TuneConfig> configs;
  
  // M分块选项
  for (auto m_tile : {16, 32, 64, 128}) {
    if (m_tile > M) continue;
    
    // N分块选项
    for (auto n_tile : {16, 32, 64, 128}) {
      if (n_tile > N) continue;
      
      // K分块选项
      for (auto k_tile : {16, 32, 64}) {
        if (k_tile > K) continue;
        
        configs.push_back({m_tile, n_tile, k_tile, 0, true});
      }
    }
  }
  
  return configs;
}
```

### 9.3 Profiling分析

```cpp
void RunWithProfiling(Kernel &kernel, uint64_t num_runs) {
  // 创建Profiling会话
  MsprofSession session;
  
  for (uint64_t i = 0; i < num_runs; ++i) {
    session.Start();
    kernel.Launch(stream);
    session.Stop();
  }
  
  // 分析结果
  auto stats = session.GetStats();
  std::cout << "Min time: " << stats.min_time << " us" << std::endl;
  std::cout << "Max time: " << stats.max_time << " us" << std::endl;
  std::cout << "Avg time: " << stats.avg_time << " us" << std::endl;
}
```

---

## 十、性能优化总结

### 10.1 优化检查清单

| 优化项 | 检查点 |
|-------|-------|
| Tiling | Tile大小是否充分利用UB？ |
| 并行 | 核心利用率是否均衡？ |
| 内存 | 活跃变量峰值是否最小？ |
| 通信 | 通信与计算是否重叠？ |
| 访存 | 是否有冗余的GM访问？ |

### 10.2 性能指标

```cpp
struct PerfMetrics {
  uint64_t compute_time;    // 计算时间
  uint64_t memory_time;     // 内存访问时间
  uint64_t comm_time;       // 通信时间
  uint64_t total_time;      // 总时间
  
  float compute_intensity;  // 计算强度 (FLOPS/Byte)
  float parallel_efficiency;// 并行效率
  float memory_efficiency;  // 内存效率
};
```

---

## 十一、总结

本文详细解析了DVM的性能优化策略：

1. **Tiling策略**：Tile大小计算、维度Tiling、MatMul Tiling
2. **UB内存管理**：布局规划、内存复用、空闲列表管理
3. **活跃变量分析**：峰值计算、优化重排
4. **多核并行**：Block维度计算、负载均衡
5. **通信优化**：AllReduce Tiling、通信计算重叠
6. **性能调优**：Tuning框架、配置搜索、Profiling分析

合理的Tiling策略和高效的内存管理是DVM实现高性能的关键。

---

## 参考资料

- [kernel.cc](../src/kernel.cc)
- [tuning.h](../src/tuning.h)
- [tuning.cc](../src/tuning.cc)
- [pass.cc](../src/pass.cc)
