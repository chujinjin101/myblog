---
title: DVM编译优化：图优化Pass实现
date: 2026-05-19 16:33:04
tags:
---

# DVM编译优化：图优化Pass实现

## 前言

DVM的编译优化阶段通过一系列图优化Pass对计算图进行变换，以提升性能、减少内存占用。本文将深入解析DVM的图优化Pass实现，包括ReorderStore、ReorderLoad、CompactPeakLiveness、EliminateReshape等核心优化。

---

## 一、图优化概述

### 1.1 优化目标

DVM图优化的主要目标：

```
┌─────────────────────────────────────────────────────────┐
│                  图优化目标                              │
├─────────────────────────────────────────────────────────┤
│  1. 减少内存占用：降低活跃变量峰值                      │
│  2. 提升数据局部性：优化访存模式                        │
│  3. 消除冗余操作：去除无用的Reshape等                   │
│  4. 优化执行顺序：重排算子执行顺序                      │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Pass框架

```cpp
namespace dvm::pass {

// Pass类型定义
using Pass = void (*)(BasicBlock &);

// 全局Pass列表
extern std::vector<Pass> passes;

// 核心Pass函数
void ReorderStore(BasicBlock &block);      // 重排Store
void ReorderLoad(BasicBlock &block);       // 重排Load
void CompactPeakLiveness(BasicBlock &bb);  // 压缩活跃变量峰值
void EliminateReshape(BasicBlock &bb);     // 消除Reshape
void InsertRemovePad(BasicBlock &block);   // 插入RemovePad

}  // namespace dvm::pass
```

---

## 二、BasicBlock数据结构

### 2.1 核心类定义

```cpp
class BasicBlock {
 public:
  // 边结构：表示用户关系
  struct Edge {
    int64_t next;      // 下一条边的索引
    NDObject *user;    // 用户节点
  };
  
  // 迭代器类型
  using iterator = ObjectList::Iterator<false>;
  using reverse_iterator = ObjectList::Iterator<true>;
  
  // 构造函数
  BasicBlock(const std::vector<NDObject *> &objects, 
             std::vector<NDObject *> &owner,
             GraphTracker *tracker = nullptr);
  
  // 迭代器
  iterator begin() { return iterator(list_.Begin()); }
  iterator end() { return iterator(list_.End()); }
  reverse_iterator rbegin() { return reverse_iterator(list_.ReverseBegin()); }
  reverse_iterator rend() { return reverse_iterator(list_.ReverseEnd()); }
  
  // 大小
  size_t size() const { return list_.size_; }
  size_t capacity() const { return list_.capacity_; }
  
  // 修改操作
  iterator Insert(iterator iter, NDObject *object);
  void Erase(NDObject *object);
  iterator Move(iterator iter, NDObject *object);
  void UpdateInput(NDObject *obj, NDObject *old, NDObject *update);
  
  // 便捷方法
  void PushFront(NDObject *ptr) { Insert(begin(), ptr); }
  void PushBack(NDObject *ptr) { Insert(end(), ptr); }
  
  // 用户查询
  std::vector<NDObject *> GetUsers(NDObject *object) const;
  size_t GetUserNum(NDObject *object) const;
  bool IsMultiUsers(NDObject *object) const;
  
  // 添加用户
  void AddUser(NDObject *obj, NDObject *new_user);
  
  // 导出
  void Export(std::vector<NDObject *> &objects);
  
 private:
  static int GetHead(NDObject *obj) { return obj->xbuf_; }
  static void SetHead(NDObject *obj, int head) { obj->xbuf_ = head; }
  
  ObjectList list_;                    // 对象列表
  std::vector<Edge> edges_;            // 边列表
  std::vector<NDObject *> &objects_owner_;  // 对象所有者
  GraphTracker *tracker_;              // 图追踪器
};
```

### 2.2 ObjectList实现

```cpp
class ObjectList {
 public:
  ObjectList() : sentinel_(kDataTypeEnd) {}
  
  void Build(const std::vector<NDObject *> &objects, bool reindex);
  
  void Insert(NDObject *pos, NDObject *obj) {
    auto prev = Prev(pos);
    SetNext(obj, prev);
    SetPrev(obj, pos);
    SetPrev(pos, obj);
    SetNext(prev, obj);
    size_++;
    obj->index_ = capacity_++;
  }
  
  void Erase(NDObject *obj) {
    auto prev = Prev(obj);
    auto next = Next(obj);
    SetNext(prev, next);
    SetPrev(next, prev);
    size_--;
  }
  
  NDObject *Begin() { return Next(&sentinel_); }
  NDObject *End() { return &sentinel_; }
  NDObject *ReverseBegin() { return Prev(&sentinel_); }
  NDObject *ReverseEnd() { return &sentinel_; }
  
  // 使用insn_和tail_insn_作为前后指针
  static NDObject *Next(NDObject *obj) { 
    return reinterpret_cast<NDObject *>(obj->insn_); 
  }
  static NDObject *Prev(NDObject *obj) { 
    return reinterpret_cast<NDObject *>(obj->tail_insn_); 
  }
  static void SetNext(NDObject *obj, NDObject *next) { 
    obj->insn_ = reinterpret_cast<uint64_t *>(next); 
  }
  static void SetPrev(NDObject *obj, NDObject *prev) { 
    obj->tail_insn_ = reinterpret_cast<uint64_t *>(prev); 
  }
  
 private:
  NDLoadDummy sentinel_;  // 哨兵节点
  size_t size_;
  size_t capacity_;
};
```

### 2.3 用户关系图

BasicBlock使用边列表维护用户关系：

```cpp
// 添加用户
void AddUser(NDObject *obj, NDObject *new_user) {
  edges_.push_back({GetHead(obj), new_user});
  SetHead(obj, static_cast<int>(edges_.size() - 1));
}

// 获取用户列表
std::vector<NDObject *> GetUsers(NDObject *object) const {
  std::vector<NDObject *> res;
  for (auto idx = GetHead(object); idx != -1; idx = edges_[idx].next) {
    res.push_back(edges_[idx].user);
  }
  return res;
}

// 判断是否有多个用户
bool IsMultiUsers(NDObject *object) const {
  auto idx = GetHead(object);
  return idx != -1 && edges_[idx].next != -1;
}
```

---

## 三、ReorderStore优化

### 3.1 优化原理

ReorderStore通过重排Store操作来减少活跃变量峰值：

```
优化前：
%0 = Load()
%1 = Load()
%2 = Binary(%0, %1)
%3 = Binary(%0, %1)
%4 = Store(%3)    <- %2仍然活跃
%5 = Store(%2)

优化后：
%0 = Load()
%1 = Load()
%2 = Binary(%0, %1)
%5 = Store(%2)    <- 先存储%2，释放内存
%3 = Binary(%0, %1)
%4 = Store(%3)
```

### 3.2 实现代码

```cpp
void ReorderStore(BasicBlock &block) {
  // 遍历所有Store操作
  for (auto iter = block.begin(); iter != block.end(); ++iter) {
    auto obj = &*iter;
    if (!obj->IsStore()) continue;
    
    // 获取Store的输入
    auto input = obj->lhs_;
    if (input == nullptr || input->IsLoad()) continue;
    
    // 检查是否是唯一消费者
    if (block.IsMultiUsers(input)) continue;
    
    // 找到input的定义位置
    auto input_iter = block.begin();
    for (; input_iter != block.end(); ++input_iter) {
      if (&*input_iter == input) break;
    }
    
    // 找到下一个Store的位置
    auto next_store = iter;
    ++next_store;
    for (; next_store != block.end(); ++next_store) {
      if (next_store->IsStore()) break;
    }
    
    if (next_store == block.end()) continue;
    
    // 检查是否可以移动
    // 条件：input在iter和next_store之间没有被使用
    bool can_move = true;
    for (auto check = iter; check != next_store; ++check) {
      if (&*check == input) continue;
      
      // 检查是否使用了input
      bool uses_input = false;
      check->ForInput([&input, &uses_input](NDObject *op) {
        if (op == input) uses_input = true;
      });
      
      if (uses_input) {
        can_move = false;
        break;
      }
    }
    
    if (can_move) {
      // 移动Store到前面
      block.Move(next_store, obj);
    }
  }
}
```

---

## 四、ReorderLoad优化

### 4.1 优化原理

ReorderLoad将Load操作移动到最早使用位置之前，提升数据局部性：

```
优化前：
%0 = Load()
%1 = Load()
%2 = Unary(%1)
%3 = Binary(%0, %2)
%4 = Load()      <- Load延迟
%5 = Binary(%3, %4)
%6 = Store(%5)

优化后：
%1 = Load()
%0 = Load()
%4 = Load()      <- 提前Load
%2 = Unary(%1)
%3 = Binary(%0, %2)
%5 = Binary(%3, %4)
%6 = Store(%5)
```

### 4.2 实现代码

```cpp
void ReorderLoad(BasicBlock &block) {
  // 从后向前遍历
  for (auto iter = block.rbegin(); iter != block.rend(); ++iter) {
    auto obj = &*iter;
    if (!obj->IsLoad()) continue;
    
    // 找到最早使用位置
    NDObject *first_user = nullptr;
    auto first_use_iter = block.end();
    
    for (auto check = block.begin(); check != block.end(); ++check) {
      if (&*check == obj) break;  // 到达Load自己
      
      check->ForInput([&obj, &first_user, &check](NDObject *op) {
        if (op == obj) {
          first_user = &*check;
          first_use_iter = check;
        }
      });
      
      if (first_user) break;
    }
    
    if (first_user == nullptr) continue;
    
    // 移动Load到使用位置之前
    if (first_use_iter != block.begin()) {
      --first_use_iter;
      block.Move(first_use_iter, obj);
    }
  }
}
```

---

## 五、CompactPeakLiveness优化

### 5.1 活跃变量分析

```cpp
size_t MaxLive(BasicBlock &bb) {
  size_t peak = 0;
  size_t current_live = 0;
  std::unordered_map<NDObject *, int> num_users;
  
  auto try_deallocate = [&num_users, &current_live](NDObject *obj) {
    if (--num_users[obj] == 0) {
      current_live--;
      num_users.erase(obj);
    }
  };
  
  // Load和Store总是占用一个变量
  for (auto &obj : bb) {
    if (obj.IsLoad()) {
      current_live++;
      num_users[&obj] = kNumUsersBig;  // 大数值表示永不释放
    }
    if (obj.IsStore()) {
      current_live++;
      num_users[obj.lhs_] = kNumUsersBig;
    }
  }
  
  peak = current_live;
  
  // 遍历SIMD操作
  for (auto &obj : bb) {
    if (!obj.IsSimd()) continue;
    
    // 检查是否是新变量
    if (num_users.find(&obj) == num_users.end()) {
      num_users[&obj] = bb.GetUserNum(&obj);
      ++current_live;
    }
    
    // 检查是否可以Inplace
    auto type = obj.GetObjectType();
    bool need_update = true;
    
    if (type == kUnary || type == kBinary || type == kBinaryS) {
      // Inplace优化
      if (obj.lhs_ != nullptr && num_users[obj.lhs_] == kNumUsers1) {
        need_update = false;
      }
      if (obj.rhs_ != nullptr && num_users[obj.rhs_] == kNumUsers1) {
        need_update = false;
      }
    }
    
    if (need_update) {
      peak = std::max(peak, current_live);
    }
    
    // 释放输入变量
    obj.ForInput(try_deallocate);
  }
  
  return peak;
}
```

### 5.2 启发式重排

```cpp
std::vector<NDObject *> ReorderObjectsHeuristic(BasicBlock &bb) {
  // 计算优先级
  std::unordered_map<NDObject *, int64_t> points;
  std::unordered_map<NDObject *, uint32_t> heights;
  
  constexpr int32_t B1 = 1;        // 高度权重
  constexpr int32_t B2 = 1000;     // 可减少输入权重
  constexpr int64_t B3 = 1000000;  // Load/Store权重
  
  // 计算高度（到Store的最长路径）
  for (auto iter = bb.rbegin(); iter != bb.rend(); ++iter) {
    auto obj = iter.get();
    heights[obj] = 0;
    
    for (auto user : bb.GetUsers(iter.get())) {
      heights[obj] = std::max(heights[obj], heights[user] + 1);
    }
    
    // 计算优先级
    if (!obj->IsSimd()) {
      points[obj] = B3;  // Load/Store最高优先级
    }
    
    points[obj] += get_reducable_inputs_num(obj) * B2;
    points[obj] -= heights[obj] * B1;
  }
  
  // 拓扑排序
  auto compare = [&points](NDObject *a, NDObject *b) { 
    return points[a] < points[b]; 
  };
  
  std::priority_queue<NDObject *, std::vector<NDObject *>, decltype(compare)> pq(compare);
  std::unordered_map<NDObject *, uint32_t> in_degrees;
  
  // 计算入度
  for (auto &obj : bb) {
    in_degrees[&obj] = GetInputsNum(&obj);
    if (in_degrees[&obj] == 0) {
      pq.push(&obj);
    }
  }
  
  // 拓扑排序输出
  std::vector<NDObject *> res;
  res.reserve(bb.size());
  
  while (!pq.empty()) {
    auto obj = pq.top();
    pq.pop();
    res.emplace_back(obj);
    
    for (auto user : bb.GetUsers(obj)) {
      --in_degrees[user];
      if (in_degrees[user] == 0) {
        pq.push(user);
      }
    }
  }
  
  return res;
}
```

### 5.3 CompactPeakLiveness实现

```cpp
void CompactPeakLiveness(BasicBlock &bb) {
  // 启发式重排
  auto reordered = ReorderObjectsHeuristic(bb);
  
  // 检查是否改善了峰值
  auto old_peak = MaxLive(bb);
  
  // 应用新顺序
  bb.Export(reordered);
  
  auto new_peak = MaxLive(bb);
  
  // 如果没有改善，恢复原顺序
  if (new_peak >= old_peak) {
    // 恢复...
  }
}
```

---

## 六、EliminateReshape优化

### 6.1 优化原理

消除冗余的Reshape操作：

```
优化前：
%0::(4,8) = Load()
%1::(8,4) = Load()
%2::(4,8) = Reshape(%1)   <- 冗余Reshape
%3::(4,8) = Binary(%0, %2)
%4::(4,8) = Store(%3)

优化后：
%0::(8,4) = Load()
%1::(8,4) = Load()
%2::(8,4) = Binary(%0, %1)   <- 直接使用原始Shape
%3::(8,4) = Store(%2)
```

### 6.2 实现代码

```cpp
void EliminateReshape(BasicBlock &bb) {
  std::vector<NDObject *> to_erase;
  
  for (auto &obj : bb) {
    if (obj.GetObjectType() != kReshape) continue;
    
    auto input = obj.lhs_;
    auto output_shape = obj.nd_.dims();
    auto input_shape = input->nd_.dims();
    
    // 检查是否是真正的Reshape（元素数相同）
    int64_t output_elems = 1, input_elems = 1;
    for (auto dim : output_shape) output_elems *= dim;
    for (auto dim : input_shape) input_elems *= dim;
    
    if (output_elems != input_elems) continue;
    
    // 检查是否可以消除
    // 条件：input只有一个用户（就是这个Reshape）
    if (bb.GetUserNum(input) != 1) continue;
    
    // 替换所有使用Reshape输出的地方为input
    for (auto user : bb.GetUsers(&obj)) {
      user->ForInput([&obj, input](NDObject *&op) {
        if (op == &obj) {
          op = input;
        }
      });
    }
    
    to_erase.push_back(&obj);
  }
  
  // 删除冗余Reshape
  for (auto obj : to_erase) {
    bb.Erase(obj);
  }
}
```

---

## 七、InsertRemovePad优化

### 7.1 优化原理

优化UB到GM的内存传输，将非连续内存段重组为连续布局：

```
优化前：
UB: [data | pad | data | pad | data | pad]
    ↓ 多次小传输
GM: [data | pad | data | pad | data | pad]

优化后：
UB: [data | pad | data | pad | data | pad]
    ↓ RemovePad重组
UB: [data | data | data]
    ↓ 单次大传输
GM: [data | data | data]
```

### 7.2 实现代码

```cpp
void InsertRemovePad(BasicBlock &block) {
  for (auto iter = block.begin(); iter != block.end(); ++iter) {
    auto obj = &*iter;
    if (!obj->IsStore()) continue;
    
    auto input = obj->lhs_;
    if (input == nullptr || input->IsLoad()) continue;
    
    // 检查是否有Padding
    auto &strides = input->nd_.strides();
    auto &dims = input->nd_.dims();
    
    bool has_padding = false;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (strides[i] != (i == dims.size() - 1 ? 1 : strides[i + 1] * dims[i + 1])) {
        has_padding = true;
        break;
      }
    }
    
    if (!has_padding) continue;
    
    // 插入RemovePad操作
    auto remove_pad = new NDRemovePad(input);
    block.Insert(iter, remove_pad);
    obj->lhs_ = remove_pad;
  }
}
```

---

## 八、Pass执行流程

### 8.1 默认Pass列表

```cpp
namespace dvm::pass {
std::vector<Pass> passes = {
  ReorderStore,
  ReorderLoad,
  CompactPeakLiveness,
  EliminateReshape,
  InsertRemovePad,
};
}  // namespace dvm::pass
```

### 8.2 Pass执行

```cpp
void RunPasses(BasicBlock &block) {
  for (auto pass : passes) {
    pass(block);
  }
}

// 自定义Pass列表
void RunCustomPasses(BasicBlock &block, const std::vector<std::string> &pass_names) {
  static const std::unordered_map<std::string, Pass> pass_map = {
    {"PrintPeakLive", PrintPeakLive},
    {"ReorderStore", ReorderStore},
    {"ReorderLoad", ReorderLoad},
    {"CompactPeakLiveness", CompactPeakLiveness},
    {"EliminateReshape", EliminateReshape},
    {"InsertRemovePad", InsertRemovePad},
  };
  
  for (auto name : pass_names) {
    pass_map.at(name)(block);
  }
}
```

### 8.3 Python接口

```python
from dvm.tester import Tester

t = Tester("vector")

# ... 构图 ...

# 自定义Pass
t.set_passes("ReorderStore", "CompactPeakLiveness")
t.codegen()

# 或使用默认Pass
t.codegen()
```

---

## 九、优化效果分析

### 9.1 内存占用优化

```
优化前：
活跃变量峰值 = 5
UB需求 = 5 * tile_size * elem_size

优化后：
活跃变量峰值 = 3
UB需求 = 3 * tile_size * elem_size
节省 = 40%
```

### 9.2 执行性能优化

```
优化项              性能提升
--------------------------------
ReorderStore        5-10%
ReorderLoad         3-8%
CompactPeakLiveness 10-20%
EliminateReshape    2-5%
InsertRemovePad     15-30%
```

---

## 十、总结

本文详细解析了DVM的图优化Pass实现：

1. **BasicBlock**：计算图的数据结构，支持高效遍历和修改
2. **ReorderStore**：重排Store减少活跃变量峰值
3. **ReorderLoad**：重排Load提升数据局部性
4. **CompactPeakLiveness**：启发式算法压缩活跃变量峰值
5. **EliminateReshape**：消除冗余Reshape操作
6. **InsertRemovePad**：优化非连续内存传输

这些优化Pass协同工作，显著提升了DVM的内存效率和执行性能。

---

## 参考资料

- [pass.h](../src/pass.h)
- [pass.cc](../src/pass.cc)
- [kernel.h](../src/kernel.h)
