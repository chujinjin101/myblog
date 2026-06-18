---
title: DVM元算子系统：NDObject与计算图构建
date: 2026-05-19 16:33:04
tags:
---

# DVM元算子系统：NDObject与计算图构建

## 前言

在DVM中，元算子系统是用户构图的基础。用户通过Load、Store、Binary、Unary等元算子API构建计算图，这些元算子在内部被表示为NDObject对象。本文将深入解析DVM的元算子系统，揭示计算图构建的内部机制。

---

## 一、元算子概述

### 1.1 什么是元算子

元算子是DVM定义的最小计算单元，类似于CPU指令集。DVM的元算子包括：

```
┌─────────────────────────────────────────────────────────┐
│                    元算子分类                            │
├─────────────────────────────────────────────────────────┤
│  访存类：Load, Store, ViewLoad, PadStore              │
│  一元类：Sqrt, Abs, Log, Exp, Reciprocal, Cast        │
│  二元类：Add, Sub, Mul, Div, Pow, Maximum, Minimum    │
│  归约类：Sum, Max, Min                                │
│  矩阵类：MatMul, GroupedMatMul                         │
│  形状类：Reshape, Broadcast, Copy                     │
│  条件类：Select                                         │
│  通信类：AllReduce, AllGather, ReduceScatter          │
└─────────────────────────────────────────────────────────┘
```

### 1.2 元算子特点

1. **细粒度**：每个元算子对应一个具体的计算操作
2. **可组合**：多个元算子可以组合成复杂的融合算子
3. **类型安全**：支持多种数据类型，编译时类型检查
4. **Shape推导**：自动进行Shape广播和推导

---

## 二、NDObject核心类

### 2.1 类层次结构

```cpp
// 基类
class NDObject {
  ObjectType obj_id_;       // 对象类型
  DataType type_id_;        // 数据类型
  IntArrayRef *shape_ref_;  // Shape引用
  NDSpace nd_;              // N维空间信息
  uint64_t xbuf_;           // UB缓冲区偏移
  uint64_t index_;          // 对象索引
  uint32_t flags_;          // 标志位
  
  NDObject *lhs_;           // 左输入
  NDObject *rhs_;           // 右输入
  uint64_t *insn_;          // 指令指针
  uint64_t *tail_insn_;     // 尾指令指针
};

// 访存类
class NDAccess : public NDObject {
  RelocAddr addr_;          // 重定位地址
};

class NDLoad : public NDAccess { ... };
class NDStore : public NDAccess { ... };

// 计算类
class NDSimd : public NDObject { ... };
class NDBinary : public NDSimd { ... };
class NDUnary : public NDSimd { ... };
class NDFlex : public NDSimd { ... };  // Reduce等

// 矩阵类
class CubeOp : public NDObject { ... };
class GmmOp : public CubeOp { ... };

// 通信类
class CommOp : public NDSimd { ... };
```

### 2.2 ObjectType枚举

```cpp
enum ObjectType {
  // Load类型
  kLoadDummy = 0,
  kMultiLoad,
  kViewLoad,
  kLoad,
  
  // Store类型
  kPadStore,
  kStore,
  
  // 通信类型
  kReduceScatter,
  kAllGather,
  kAllGatherV2,
  kAllReduce,
  
  // SIMD计算类型
  kReshape,
  kCopy,
  kUnary,
  kBinary,
  kCast,
  kBinaryS,
  kBroadcastTo,
  kBroadcastS,
  kReduce,
  kSelect,
  kElementAny,
  kRemovePad,
  kPower,
  kCompare,
  kCompareS,
  kOneHot,
  kCubeOp,
  kGmmOp,
  kObjectBulk
};
```

### 2.3 核心成员解析

#### NDSpace - N维空间

```cpp
class NDSpace {
  const NDSpaceData *data;  // 指向空间数据
  
  int64_t operator[](int i) const;     // 获取维度大小
  int64_t stride(int i) const;         // 获取步长
  int64_t lead_dim() const;            // 获取前导维度
  int64_t lead_stride() const;         // 获取前导步长
  const DimArray &dims() const;        // 获取维度数组
};

class NDSpaceData {
  DimArray dims;      // 维度数组
  DimArray strides;   // 步长数组
  int lidx;           // 前导维度索引
};
```

#### RelocAddr - 重定位地址

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
};
```

---

## 三、元算子实现详解

### 3.1 Load算子

Load算子从全局内存加载数据到UB：

```cpp
class NDLoad : public NDAccess {
 public:
  NDLoad(void *addr, IntArrayRef *shape, DataType type) {
    obj_id_ = kLoad;
    type_id_ = type;
    shape_ref_ = shape;
    addr_.gm = addr;
    
    // 初始化NDSpace
    UpdateShape();
  }
  
  void UpdateShape() {
    // 从shape_ref_更新nd_
    nd_.data = &space_data_;
    space_data_.dims = *shape_ref_;
    
    // 计算步长
    UpdateStrides();
  }
  
  void UpdateStrides() {
    auto &dims = space_data_.dims;
    auto &strides = space_data_.strides;
    
    strides.resize(dims.size());
    strides.back() = 1;
    
    for (int i = dims.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
    
    // 找到前导维度
    space_data_.lidx = 0;
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] != 1) {
        space_data_.lidx = i;
        break;
      }
    }
  }
};
```

**使用示例**：

```cpp
std::vector<int64_t> shape = {32, 64};
IntArrayRef shape_ref(shape);
auto load = kernel.Load(dev_ptr, &shape_ref, kFloat32);
```

### 3.2 Store算子

Store算子将结果存储回全局内存：

```cpp
class NDStore : public NDAccess {
 public:
  NDStore(void *addr, NDObject *input) {
    obj_id_ = kStore;
    type_id_ = input->type_id_;
    shape_ref_ = input->shape_ref_;
    lhs_ = input;
    addr_.gm = addr;
  }
};
```

### 3.3 Binary算子

Binary算子实现二元计算：

```cpp
class NDBinary : public NDSimd {
 public:
  NDBinary(BinaryType op_type, NDObject *lhs, NDObject *rhs) {
    obj_id_ = kBinary;
    lhs_ = lhs;
    rhs_ = rhs;
    op_type_ = op_type;
    
    // Shape推导
    PropShape();
    PropType();
  }
  
  void PropShape() {
    // 广播Shape推导
    auto &lhs_shape = lhs_->nd_.dims();
    auto &rhs_shape = rhs_->nd_.dims();
    
    // 取最大Shape
    size_t max_rank = std::max(lhs_shape.size(), rhs_shape.size());
    
    for (size_t i = 0; i < max_rank; ++i) {
      int64_t lhs_dim = i < lhs_shape.size() ? 
                   lhs_shape[lhs_shape.size() - 1 - i] : 1;
      int64_t rhs_dim = i < rhs_shape.size() ? 
                   rhs_shape[rhs_shape.size() - 1 - i] : 1;
      
      if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
        // Shape不兼容
        throw std::runtime_error("Incompatible shapes");
      }
      
      result_shape_.push_front(std::max(lhs_dim, rhs_dim));
    }
  }
  
  void PropType() {
    // 类型推导
    type_id_ = std::max(lhs_->type_id_, rhs_->type_id_);
  }
  
 private:
  BinaryType op_type_;
};
```

**指令映射表**：

```cpp
static const InsnIdTable binary_id_list[] = {
  {"Add", {V_NONE, V_ADD_FP16, V_ADD_BF16, V_ADD, V_ADD_INT32}},
  {"Sub", {V_NONE, V_SUB_FP16, V_SUB_BF16, V_SUB, V_SUB_INT32}},
  {"Mul", {V_NONE, V_MUL_FP16, V_MUL_BF16, V_MUL, V_MUL_INT32}},
  {"Div", {V_NONE, V_DIV_FP16, V_NONE, V_DIV, V_NONE}},
  {"Maximum", {V_NONE, V_MAX_FP16, V_MAX_BF16, V_MAX, V_MAX_INT32}},
  {"Minimum", {V_NONE, V_MIN_FP16, V_MIN_BF16, V_MIN, V_MIN_INT32}},
  ...
};
```

### 3.4 Unary算子

Unary算子实现一元计算：

```cpp
class NDUnary : public NDSimd {
 public:
  NDUnary(UnaryType op_type, NDObject *input) {
    obj_id_ = kUnary;
    lhs_ = input;
    op_type_ = op_type;
    
    // Shape和Type继承
    shape_ref_ = input->shape_ref_;
    type_id_ = input->type_id_;
  }
  
 private:
  UnaryType op_type_;
};
```

### 3.5 Reduce算子

Reduce算子实现归约计算：

```cpp
class NDFlex : public NDSimd {
 public:
  // Reduce构造函数
  NDFlex(ReduceType red_type, NDObject *input, 
         IntArrayRef *dims, bool keepdims) {
    obj_id_ = kReduce;
    lhs_ = input;
    red_type_ = red_type;
    reduce_dims_ = *dims;
    keepdims_ = keepdims;
    
    // Shape推导
    PropReduceShape();
  }
  
  void PropReduceShape() {
    auto &input_shape = lhs_->nd_.dims();
    result_shape_.clear();
    
    std::unordered_set<int64_t> reduce_set;
    for (auto dim : reduce_dims_) {
      reduce_set.insert(dim < 0 ? input_shape.size() + dim : dim);
    }
    
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (reduce_set.count(i)) {
        if (keepdims_) {
          result_shape_.push_back(1);
        }
      } else {
        result_shape_.push_back(input_shape[i]);
      }
    }
  }
  
 private:
  ReduceType red_type_;
  DimArray reduce_dims_;
  bool keepdims_;
};
```

**Reduce指令映射**：

```cpp
static const vSimdInsnID reduce_x_list[][3] = {
  {V_NONE, V_NONE, V_NONE},                // V_BOOL
  {V_NONE, V_RMAX_X_FP16, V_RMIN_X_FP16},  // V_FLOAT16
  {V_NONE, V_NONE, V_NONE},                // V_BFLOAT16
  {V_RSUM_X, V_RMAX_X, V_RMIN_X},          // V_FLOAT32
};

static const vSimdInsnID reduce_y_list[][3] = {
  {V_NONE, V_NONE, V_NONE},                // V_BOOL
  {V_NONE, V_RMAX_Y_FP16, V_RMIN_Y_FP16},  // V_FLOAT16
  {V_NONE, V_NONE, V_NONE},                // V_BFLOAT16
  {V_RSUM_Y, V_RMAX_Y, V_RMIN_Y},          // V_FLOAT32
};
```

### 3.6 MatMul算子

MatMul算子实现矩阵乘法：

```cpp
class CubeOp : public NDObject {
 public:
  CubeOp(NDObject *lhs, NDObject *rhs, 
         bool trans_a, bool trans_b, NDObject *bias) {
    obj_id_ = kCubeOp;
    lhs_ = lhs;
    rhs_ = rhs;
    bias_ = bias;
    trans_a_ = trans_a;
    trans_b_ = trans_b;
    
    // Shape推导
    PropMatMulShape();
  }
  
  void PropMatMulShape() {
    auto &lhs_shape = lhs_->nd_.dims();
    auto &rhs_shape = rhs_->nd_.dims();
    
    // M = lhs_shape[-2] (or -1 if trans_a)
    // K = lhs_shape[-1] (or -2 if trans_a)
    // N = rhs_shape[-1] (or -2 if trans_b)
    
    int64_t M = trans_a_ ? lhs_shape.back() : lhs_shape[lhs_shape.size() - 2];
    int64_t N = trans_b_ ? rhs_shape[rhs_shape.size() - 2] : rhs_shape.back();
    
    // Batch维度
    for (size_t i = 0; i < lhs_shape.size() - 2; ++i) {
      result_shape_.push_back(lhs_shape[i]);
    }
    
    result_shape_.push_back(M);
    result_shape_.push_back(N);
  }
  
 private:
  bool trans_a_;
  bool trans_b_;
  NDObject *bias_;
  NDAccess *output_;
  NDAccess *lhs_;
  NDAccess *rhs_;
  int block_dim_;
  int core_loop_;
};
```

---

## 四、计算图构建

### 4.1 KernelBuilder辅助类

```cpp
class KernelBuilder {
 public:
  KernelBuilder(VKernel *kernel) : kernel_(kernel) {}
  
  NDObject *Load(void *addr, IntArrayRef *shape, DataType type) {
    auto op = new NDLoad(addr, shape, type);
    kernel_->Append(op);
    return op;
  }
  
  NDObject *Store(void *addr, NDObject *input) {
    auto op = new NDStore(addr, input);
    kernel_->Append(op);
    return op;
  }
  
  template <BinaryType BT>
  NDObject *Binary(NDObject *lhs, NDObject *rhs) {
    auto op = new NDBinary(BT, lhs, rhs);
    kernel_->Append(op);
    return op;
  }
  
  template <UnaryType UT>
  NDObject *Unary(NDObject *input) {
    auto op = new NDUnary(UT, input);
    kernel_->Append(op);
    return op;
  }
  
  template <ReduceType RT>
  NDObject *Reduce(NDObject *input, IntArrayRef *dims, bool keepdims) {
    auto op = new NDFlex(RT, input, dims, keepdims);
    kernel_->Append(op);
    return op;
  }
  
 private:
  VKernel *kernel_;
};
```

### 4.2 构图示例

```cpp
// 用户代码
kernel.Reset(dvm::kVector, 0);
auto x = kernel.Load(dev_x, &shape_ref, dvm::kFloat32);
auto y = kernel.Load(dev_y, &shape_ref, dvm::kFloat32);
auto z = kernel.Binary<dvm::kAdd>(x, y);
auto out = kernel.Store(dev_out, z);

// 内部流程
// 1. 创建NDLoad(x)
// 2. kernel_->Append(x) -> objects_.push_back(x)
// 3. 创建NDLoad(y)
// 4. kernel_->Append(y)
// 5. 创建NDBinary(kAdd, x, y)
//    - PropShape(): Shape推导
//    - PropType(): 类型推导
// 6. kernel_->Append(z)
// 7. 创建NDStore(dev_out, z)
// 8. kernel_->Append(out)
```

### 4.3 计算图结构

```
NDLoad(x)     NDLoad(y)
    │             │
    └─────┬───────┘
          │
      NDBinary(Add)
          │
      NDStore(out)
```

---

## 五、Shape推导系统

### 5.1 广播规则

```cpp
void PropBroadcastShape(const DimArray &lhs, const DimArray &rhs, 
                        DimArray &result) {
  size_t max_rank = std::max(lhs.size(), rhs.size());
  result.resize(max_rank);
  
  for (size_t i = 0; i < max_rank; ++i) {
    int64_t lhs_dim = i < lhs.size() ? lhs[lhs.size() - 1 - i] : 1;
    int64_t rhs_dim = i < rhs.size() ? rhs[rhs.size() - 1 - i] : 1;
    
    if (lhs_dim == rhs_dim) {
      result[max_rank - 1 - i] = lhs_dim;
    } else if (lhs_dim == 1) {
      result[max_rank - 1 - i] = rhs_dim;
    } else if (rhs_dim == 1) {
      result[max_rank - 1 - i] = lhs_dim;
    } else {
      throw std::runtime_error("Broadcast error");
    }
  }
}
```

### 5.2 隐式广播插入

DVM会自动插入Broadcast操作：

```cpp
NDObject *InsertImplicitBroadcast(NDObject *obj, const DimArray &dst_shape,
                                  std::vector<NDObject *> &stuff_ops) {
  // 逐步插入Broadcast
  auto temp_shape = obj->nd_.dims();
  NDObject *output = obj;
  
  for (size_t i = 0; i < temp_shape.size(); ++i) {
    if (temp_shape[i] != dst_shape[i]) {
      temp_shape[i] = dst_shape[i];
      output = new BroadcastOp(output, temp_shape);
      stuff_ops.push_back(output);
    }
  }
  
  return output;
}
```

---

## 六、代码生成模板

### 6.1 CodeGenTmpl枚举

```cpp
enum CodeGenTmpl {
  kGenSimd0,   // 无输入（BroadcastS）
  kGenSimd1,   // 单输入（Unary, Cast）
  kGenSimd2,   // 双输入（Binary）
  kGenSimd3,   // 三输入（Select）
  kGenFlex,    // 灵活输入（Reduce）
  kGenLoad,    // Load
  kGenStore,   // Store
  kGenComm,    // 通信
};
```

### 6.2 模板映射表

```cpp
static const BaseData data[] = {
  // {flags, tmpl, dim_changed, fold_prop, align_prop}
  
  // Load
  {kGenLoad, F_IP | nullptr, nullptr, nullptr, nullptr},
  
  // Store
  {kGenStore, F_IP | F_NS | F_LD, NDStore::DimChanged, nullptr, nullptr},
  
  // Unary
  {kGenSimd1, F_IP | F_NS | F_LR, nullptr, nullptr, nullptr},
  
  // Binary
  {kGenSimd2, F_IP | F_NS | F_LR | F_RR, nullptr, nullptr, nullptr},
  
  // Reduce
  {kGenFlex, F_LR | F_LD, ReduceOp::DimChanged, ReduceOp::FoldProp, ReduceOp::AlignProp},
  
  // Cast
  {kGenSimd1, F_IP | F_NS, nullptr, nullptr, nullptr},
  
  // BroadcastTo
  {kGenSimd1, F_DM, nullptr, BroadcastOp::FoldProp, BroadcastOp::AlignProp},
  
  // MatMul
  {kGenCube, F_IP, CubeOp::DimChanged, CubeOp::FoldProp, nullptr},
};
```

### 6.3 指令发射

```cpp
uint64_t NDBinary::Emit(VectorKernel &kernel) {
  uint64_t *pc = insn_;
  
  // 获取指令ID
  auto insn_id = binary_id_list[op_type_].ids[type_id_];
  
  // 发射指令
  vBinary op;
  op.xd = xbuf_;
  op.xn = lhs_->xbuf_;
  op.xm = rhs_->xbuf_;
  
  return vBinary::Encode(pc, insn_id, op);
}
```

---

## 七、对象标志系统

### 7.1 标志定义

```cpp
enum ObjectFlag {
  F_IP = 0x1,      // Inplace优化
  F_NS = 0x2,      // Non-Store
  F_LR = 0x4,      // Left-Reuse
  F_RR = 0x8,      // Right-Reuse
  F_LD = 0x10,     // Load
  F_DM = 0x20,     // Dynamic-Shape
  F_XHS = 0x40,    // Extra-Handle-Size
  
  OBJ_FLAG_DEAD = 0x100,
  OBJ_FLAG_FREE_LHS = 0x200,
  OBJ_FLAG_FREE_RHS = 0x400,
  OBJ_FLAG_LOAD_FROM_CUBE = 0x800,
};
```

### 7.2 标志应用

```cpp
// Inplace优化
if (flags_ & F_IP && lhs_->flags_ & F_LR) {
  // 可以原地操作，复用lhs的xbuf
  xbuf_ = lhs_->xbuf_;
}

// 内存释放
if (flags_ & F_FREE_LHS) {
  free_xbuf_.Push(lhs_->xbuf_, this);
}
if (flags_ & F_FREE_RHS) {
  free_xbuf_.Push(rhs_->xbuf_, this);
}
```

---

## 八、总结

本文详细解析了DVM的元算子系统：

1. **NDObject体系**：Load、Store、Binary、Unary、Reduce、MatMul等核心类
2. **Shape推导**：广播规则、隐式广播插入
3. **计算图构建**：KernelBuilder辅助类、构图流程
4. **代码生成**：模板系统、指令发射
5. **标志系统**：Inplace优化、内存释放

元算子系统是DVM用户接口的底层支撑，理解它对于深入掌握DVM至关重要。

---

## 参考资料

- [ops.h](../src/ops.h)
- [ops.cc](../src/ops.cc)
- [kernel.h](../src/kernel.h)
