---
title: DVM Python接口详解与实战
date: 2026-05-19 16:33:04
tags:
---

# DVM Python接口详解与实战

## 前言

在上一篇《DVM快速入门：从Hello World到自定义算子》中，我们初步体验了DVM的基本使用方式。本文将深入剖析DVM的Python接口实现原理，详细介绍各个API的使用方法，并通过实战案例帮助你掌握DVM Python接口的高级用法。

---

## 一、Python接口架构

### 1.1 整体架构

DVM的Python接口采用分层设计：

```
┌─────────────────────────────────────────────────────────┐
│                    用户代码层                             │
│  @dvm.kernel 装饰器、JitKernel、Tester                   │
├─────────────────────────────────────────────────────────┤
│                    Python封装层                          │
│  __init__.py, jit.py, tester.py                         │
├─────────────────────────────────────────────────────────┤
│                    pybind11绑定层                        │
│  _dvm_py.so, pybind_api.cc                              │
├─────────────────────────────────────────────────────────┤
│                    C++核心层                             │
│  Kernel, NDObject, VKernel                              │
└─────────────────────────────────────────────────────────┘
```

### 1.2 核心模块

| 模块 | 文件 | 功能 |
|-----|------|------|
| dvm | __init__.py | 模块入口，导出公共API |
| dvm.jit | jit.py | JIT编译装饰器实现 |
| dvm.tester | tester.py | 测试工具类 |
| dvm._dvm_py | _dvm_py.so | pybind11绑定库 |

### 1.3 模块初始化流程

```python
# dvm/__init__.py
from . import _dvm_py as _core
from ._dvm_py import DataType, Device, Kernel, PyKernel, NDObject, IntArrayRef, ScalarRef
from .jit import kernel

# 导出数据类型常量
bool_ = DataType.bool
float16 = DataType.float16
bfloat16 = DataType.bfloat16
float32 = DataType.float32
int32 = DataType.int32
int64 = DataType.int64
```

---

## 二、核心类详解

### 2.1 DataType - 数据类型

DVM支持以下数据类型：

```python
import dvm

# 数据类型常量
dvm.bool_      # 布尔类型
dvm.float16    # 半精度浮点 (FP16)
dvm.bfloat16   # Brain Float16 (BF16)
dvm.float32    # 单精度浮点 (FP32)
dvm.int32      # 32位整数
dvm.int64      # 64位整数
```

**类型特性**：

| 类型 | 字节数 | 适用场景 |
|-----|-------|---------|
| float16 | 2 | 推理加速、显存优化 |
| bfloat16 | 2 | 训练场景、数值稳定性 |
| float32 | 4 | 高精度计算、Reduce操作 |
| int32 | 4 | 索引、整数运算 |

### 2.2 NDObject - N维张量对象

`NDObject`是DVM中表示张量的核心类：

```python
class NDObject:
    def shape(self) -> Tuple[int, ...]:
        """获取张量形状"""
        pass
    
    def dtype(self) -> DataType:
        """获取数据类型"""
        pass
```

**使用示例**：

```python
import dvm
import numpy as np

@dvm.kernel
def example(k, x):
    a = k.load(x, dvm.float32)
    print(f"Shape: {a.shape()}")  # 输出形状
    print(f"DType: {a.dtype()}")  # 输出类型
    return k.store(a)

x = np.random.randn(32, 64).astype(np.float32)
result = example(x)
```

### 2.3 IntArrayRef - 动态Shape引用

`IntArrayRef`用于支持动态Shape场景：

```python
class IntArrayRef:
    def __init__(self, shape: Sequence[int] = None):
        """创建Shape引用"""
        pass
    
    def shape(self) -> Tuple[int, ...]:
        """获取当前Shape"""
        pass
    
    def update(self, shape: Sequence[int]) -> None:
        """更新Shape"""
        pass
```

**动态Shape示例**：

```python
import dvm

# 创建动态Shape引用
shape_ref = dvm.IntArrayRef()

@dvm.kernel
def dynamic_kernel(k):
    x = k.load(shape_ref, dvm.float32)
    y = k.add(x, 1.0)
    return k.store(y)

# 第一次执行：Shape = [32, 64]
shape_ref.update([32, 64])
result1 = dynamic_kernel(np.ones([32, 64], dtype=np.float32))

# 第二次执行：Shape = [128, 256]
shape_ref.update([128, 256])
result2 = dynamic_kernel(np.ones([128, 256], dtype=np.float32))
```

### 2.4 ScalarRef - 动态标量引用

`ScalarRef`用于支持动态标量参数：

```python
class ScalarRef:
    def update(self, value: Union[int, float]) -> None:
        """更新标量值"""
        pass
```

**使用示例**：

```python
import dvm

@dvm.kernel
def scale_kernel(k, x, scale):
    scale_ref = k.scalar(dvm.float32)
    a = k.load(x, dvm.float32)
    b = k.mul(a, scale_ref)
    return k.store(b)

# 动态更新scale值
scale = dvm.ScalarRef()
scale.update(0.5)
result = scale_kernel(x, scale)
```

---

## 三、Kernel类详解

### 3.1 Kernel类层次

```
Kernel (基类 - pybind11绑定)
├── PyKernel (运行时Kernel)
│   └── JitKernel (JIT编译Kernel)
└── Tester (测试工具类)
```

### 3.2 Kernel核心API

#### 3.2.1 内存操作

```python
# 加载张量
def load(self, shape: ShapeLike, type: DataType) -> NDObject:
    """从全局内存加载张量"""
    pass

# 非连续张量加载
def view_load(self, shape: ShapeLike, stride: ShapeLike, type: DataType) -> NDObject:
    """加载非连续内存张量"""
    pass

# 存储张量
def store(self, obj: NDObject, type: DataType = None) -> NDObject:
    """存储结果到全局内存"""
    pass
```

#### 3.2.2 一元操作

```python
# 数学函数
def sqrt(self, input: NDObject) -> NDObject:    # 平方根
def abs(self, input: NDObject) -> NDObject:     # 绝对值
def log(self, input: NDObject) -> NDObject:     # 自然对数
def exp(self, input: NDObject) -> NDObject:     # 自然指数
def reciprocal(self, input: NDObject) -> NDObject:  # 倒数

# 取整函数
def round(self, input: NDObject) -> NDObject:   # 四舍五入
def floor(self, input: NDObject) -> NDObject:   # 向下取整
def ceil(self, input: NDObject) -> NDObject:    # 向上取整
def trunc(self, input: NDObject) -> NDObject:   # 截断

# 逻辑函数
def logical_not(self, input: NDObject) -> NDObject:  # 逻辑非
def isfinite(self, input: NDObject) -> NDObject:     # 有限判断
```

#### 3.2.3 二元操作

```python
# 算术运算
def add(self, lhs, rhs) -> NDObject:     # 加法
def sub(self, lhs, rhs) -> NDObject:     # 减法
def mul(self, lhs, rhs) -> NDObject:     # 乘法
def div(self, lhs, rhs) -> NDObject:     # 除法
def pow(self, lhs, rhs) -> NDObject:     # 幂运算

# 比较运算
def equal(self, lhs, rhs) -> NDObject:        # 等于
def not_equal(self, lhs, rhs) -> NDObject:    # 不等于
def greater(self, lhs, rhs) -> NDObject:      # 大于
def greater_equal(self, lhs, rhs) -> NDObject: # 大于等于
def less(self, lhs, rhs) -> NDObject:         # 小于
def less_equal(self, lhs, rhs) -> NDObject:   # 小于等于

# 极值运算
def maximum(self, lhs, rhs) -> NDObject:  # 最大值
def minimum(self, lhs, rhs) -> NDObject:  # 最小值

# 逻辑运算
def logical_and(self, lhs, rhs) -> NDObject:  # 逻辑与
def logical_or(self, lhs, rhs) -> NDObject:   # 逻辑或
```

**参数类型支持**：

```python
# 张量与张量
c = k.add(a, b)

# 张量与标量
c = k.add(a, 1.0)
c = k.add(a, 2)

# 标量与张量
c = k.mul(0.5, a)
```

#### 3.2.4 Reduce操作

```python
def sum(self, input: NDObject, dims: Sequence[int], keepdims: bool = False) -> NDObject:
    """求和归约"""
    pass

def max(self, input: NDObject, dims: Sequence[int], keepdims: bool = False) -> NDObject:
    """最大值归约"""
    pass

def min(self, input: NDObject, dims: Sequence[int], keepdims: bool = False) -> NDObject:
    """最小值归约"""
    pass
```

**使用示例**：

```python
import dvm

@dvm.kernel
def reduce_example(k, x):
    a = k.load(x, dvm.float32)
    
    # 对第0维求和，不保持维度
    sum_0 = k.sum(a, (0,), False)  # Shape: [d1, d2, ...]
    
    # 对第0、1维求最大值，保持维度
    max_01 = k.max(a, (0, 1), True)  # Shape: [1, 1, d2, ...]
    
    # 全局求和
    sum_all = k.sum(a, tuple(range(len(a.shape()))), False)  # Shape: []
    
    return k.store(sum_all)
```

#### 3.2.5 矩阵乘法

```python
def matmul(
    self,
    lhs: NDObject,
    rhs: NDObject,
    trans_a: bool,
    trans_b: bool,
    bias: Optional[NDObject] = None
) -> NDObject:
    """矩阵乘法"""
    pass
```

**参数说明**：
- `lhs`: 左矩阵
- `rhs`: 右矩阵
- `trans_a`: 是否转置左矩阵
- `trans_b`: 是否转置右矩阵
- `bias`: 可选的偏置向量

**使用示例**：

```python
import dvm

@dvm.kernel
def matmul_example(k, a, b, bias):
    a = k.load(a, dvm.float16)
    b = k.load(b, dvm.float16)
    bias = k.load(bias, dvm.float32)
    
    # C = A @ B^T + bias
    c = k.matmul(a, b, False, True, bias)
    
    return k.store(c)
```

#### 3.2.6 其他操作

```python
# 类型转换
def cast(self, input: NDObject, type: DataType) -> NDObject:
    pass

# 形状操作
def reshape(self, input: NDObject, shape: ShapeLike) -> NDObject:
    pass

# 广播
def broadcast(self, input: NDObject, shape: ShapeLike) -> NDObject:
    pass

# 条件选择
def select(self, cond: NDObject, lhs: NDObject, rhs: NDObject) -> NDObject:
    pass

# 拷贝
def copy(self, input: NDObject) -> NDObject:
    pass
```

---

## 四、装饰器机制详解

### 4.1 @dvm.kernel装饰器

`@dvm.kernel`是DVM最核心的装饰器，它将普通Python函数转换为可执行的Kernel：

```python
# jit.py
def kernel(ktype="split", dynamic=True):
    if callable(ktype):
        func = ktype
        kobj = JitKernel("split", True)
        kobj.build(func)
        return kobj

    def decorate(func):
        kobj = JitKernel(ktype, dynamic)
        kobj.build(func)
        return kobj

    return decorate
```

### 4.2 JitKernel类

```python
class JitKernel(Kernel):
    def __init__(self, ker_type, dynamic):
        # 解析Kernel类型和动态标志
        if dynamic:
            ker_type += ",dyn" if ":" in ker_type else ":dyn"
        
        # 获取设备ID
        dev_conf = os.getenv("DEVICE_ID")
        dev_id = int(dev_conf) if dev_conf else 0
        
        Kernel.__init__(self, ker_type, "dev", dev_id)
        self.inputs = None
        self.outputs = None
        self.dynamic = dynamic

    def build(self, func):
        # 解析函数参数数量
        arg_cnt = func.__code__.co_argcount - 1
        args = [i for i in range(arg_cnt)]
        self.inputs = [None] * arg_cnt
        
        # 执行构图函数
        self.outputs = func(self, *args)
        
        # 静态Shape提前编译
        if not self.dynamic:
            self.codegen(None)

    def __call__(self, *args):
        # 绑定输入数据
        for inp, arg in zip(self.inputs, args):
            Kernel.input(self, inp, arg)
        
        # 动态Shape需要重新编译
        if self.dynamic:
            self.codegen(None)
        
        # 执行Kernel
        self.run()
        
        # 返回输出
        if isinstance(self.outputs, (list, tuple)):
            return [self.output(x) for x in self.outputs]
        else:
            return self.output(self.outputs)
```

### 4.3 装饰器使用方式

#### 基本用法

```python
import dvm

@dvm.kernel
def my_add(k, x, y):
    a = k.load(x, dvm.float32)
    b = k.load(y, dvm.float32)
    c = k.add(a, b)
    return k.store(c)
```

#### 指定Kernel类型

```python
# Vector Kernel
@dvm.kernel(ktype="vector")
def vector_op(k, x):
    pass

# Cube Kernel (MatMul)
@dvm.kernel(ktype="cube")
def matmul_op(k, a, b):
    pass

# Mix Kernel (MatMul + Vector)
@dvm.kernel(ktype="mix")
def mix_op(k, a, b):
    pass

# Split Kernel (自动拆分)
@dvm.kernel(ktype="split")
def complex_op(k, x):
    pass
```

#### 静态Shape优化

```python
# 静态Shape：编译一次，多次执行
@dvm.kernel(dynamic=False)
def static_kernel(k, x):
    a = k.load(x, dvm.float32)
    return k.store(a)

# 动态Shape：每次执行重新编译
@dvm.kernel(dynamic=True)  # 默认值
def dynamic_kernel(k, x):
    a = k.load(x, dvm.float32)
    return k.store(a)
```

---

## 五、Tester测试工具类

`Tester`是DVM提供的测试工具类，简化了算子开发和验证流程。

### 5.1 基本使用

```python
from dvm.tester import Tester
import numpy as np

def test_add():
    t = Tester("vector")  # 创建Vector Kernel测试器
    
    # 准备输入数据
    x = np.random.randn(32, 64).astype(np.float32)
    y = np.random.randn(32, 64).astype(np.float32)
    
    # 构图
    a = t.load(x)
    b = t.load(y)
    c = t.add(a, b)
    
    # 存储并验证
    t.store_expect(c, x + y)  # 自动验证结果
    
    # 执行并检查
    assert t.run_check()
```

### 5.2 Tester核心方法

```python
class Tester(Kernel):
    def __init__(self, ker_type="", use_pass_opt=False, run_mode="dev", comm=None):
        """初始化测试器"""
        pass
    
    def load(self, shape_arr, dtype=None):
        """加载数组并自动处理类型转换"""
        pass
    
    def store_expect(self, x, e, eps=None):
        """存储并记录期望结果"""
        pass
    
    def run_check(self, verbose=False):
        """执行并验证结果"""
        pass
    
    def run_perf(self):
        """性能测试"""
        pass
    
    def run_msprof(self, path, test_num=10):
        """Profiling分析"""
        pass
```

### 5.3 性能测试

```python
from dvm.tester import Tester
import numpy as np

def perf_test():
    t = Tester("vector")
    
    x = np.random.randn(1024, 1024).astype(np.float32)
    y = np.random.randn(1024, 1024).astype(np.float32)
    
    a = t.load(x)
    b = t.load(y)
    c = t.add(a, b)
    t.store(c)
    
    # 运行性能测试
    perf = t.run_perf()
    print(perf)
    # 输出: Kernel Time Summary: perf_test
    #       min (us)       max (us)       mean (us)      
    #       12.3456        15.6789        13.4567
```

### 5.4 Profiling分析

```python
def msprof_test():
    t = Tester("vector")
    
    # ... 构图 ...
    
    # 生成Profiling数据
    result = t.run_msprof("./profiling_output", test_num=100)
    print(result)
```

---

## 六、高级特性

### 6.1 Parallel并行执行

```python
import dvm

@dvm.kernel(ktype="parallel")
def parallel_example(k, a, b, c, d):
    # 第一个并行分支：Vector计算
    k.parallel_add(dvm.Kernel.K_VEC)
    a = k.load(a, dvm.float32)
    b = k.load(b, dvm.float32)
    ab = k.add(a, b)
    out1 = k.store(ab)
    
    # 第二个并行分支：MatMul计算
    k.parallel_add(dvm.Kernel.K_CUBE)
    c = k.load(c, dvm.float16)
    d = k.load(d, dvm.float16)
    cd = k.matmul(c, d, False, False)
    out2 = k.store(cd)
    
    return [out1, out2]
```

### 6.2 Sequence顺序执行

```python
@dvm.kernel(ktype="seq")
def sequence_example(k, a, b):
    # 第一个子Kernel
    k.seq_add(dvm.Kernel.K_VEC)
    a = k.load(a, dvm.float32)
    a_cast = k.cast(a, dvm.float16)
    
    # 第二个子Kernel
    k.seq_add(dvm.Kernel.K_MIX)
    b = k.load(b, dvm.float16)
    c = k.matmul(a_cast, b, False, False)
    d = k.add(c, 1.0)
    
    return k.store(d)
```

### 6.3 通信算子

```python
from dvm.tester import Tester, CommScope

def test_allreduce():
    # 创建多进程通信域
    with CommScope(0, 1, 2, 3):  # 使用4个设备
        t = Tester(comm=...)
        
        x = np.random.randn(1024).astype(np.float32)
        a = t.load(x)
        
        # AllReduce求和
        b = t.allreduce("sum", a)
        
        t.store_expect(b, x * 4)  # 4个进程求和
        assert t.run_check()
```

### 6.4 自定义Tiling

```python
@dvm.kernel
def custom_tiling(k, x):
    a = k.load(x, dvm.float32)
    b = k.add(a, 1.0)
    
    # 自定义Tiling策略
    k.tile(start=0, end=2, num=128, factor=0)
    
    return k.store(b)
```

---

## 七、调试技巧

### 7.1 Dump计算图

```python
@dvm.kernel
def debug_kernel(k, x):
    a = k.load(x, dvm.float32)
    b = k.add(a, 1.0)
    c = k.mul(b, 2.0)
    
    # 打印计算图
    print(k.dump())
    
    return k.store(c)
```

输出示例：
```
%0 = Load([32, 64], float32)
%1 = Add(%0, 1.0)
%2 = Mul(%1, 2.0)
%3 = Store(%2)
```

### 7.2 反汇编字节码

```python
@dvm.kernel
def das_kernel(k, x):
    a = k.load(x, dvm.float32)
    b = k.add(a, 1.0)
    
    result = k.store(b)
    
    # 打印字节码
    print(k.das())
    
    return result
```

输出示例：
```
V_LOAD: addr=0x..., shape=[32, 64], type=float32
V_ADD: lhs=%0, rhs=1.0
V_STORE: addr=0x..., src=%1
```

### 7.3 Verbose模式

```python
from dvm.tester import Tester

def verbose_test():
    t = Tester("vector")
    
    # ... 构图 ...
    
    # 详细输出模式
    t.run(verbose=True)
    # 输出:
    # ******* before tiling *******
    # ******* after tiling *******
    # ********* bytecode *********
```

---

## 八、实战案例

### 8.1 LayerNorm实现

```python
import dvm
import numpy as np

@dvm.kernel
def layer_norm(k, x, gamma, beta, eps):
    x = k.load(x, dvm.float32)
    gamma = k.load(gamma, dvm.float32)
    beta = k.load(beta, dvm.float32)
    eps = k.scalar(eps)
    
    # 计算均值
    mean = k.sum(x, (-1,), True)
    mean = k.div(mean, x.shape()[-1])
    
    # 计算方差
    x_sub = k.sub(x, mean)
    var = k.mul(x_sub, x_sub)
    var = k.sum(var, (-1,), True)
    var = k.div(var, x.shape()[-1])
    
    # 归一化
    var_eps = k.add(var, eps)
    std = k.sqrt(var_eps)
    x_norm = k.div(x_sub, std)
    
    # 缩放和平移
    out = k.mul(gamma, x_norm)
    out = k.add(out, beta)
    
    return k.store(out)

# 测试
x = np.random.randn(32, 128).astype(np.float32)
gamma = np.ones(128, dtype=np.float32)
beta = np.zeros(128, dtype=np.float32)

result = layer_norm(x, gamma, beta, 1e-5)
```

### 8.2 Softmax实现

```python
@dvm.kernel
def softmax(k, x):
    x = k.load(x, dvm.float32)
    
    # 数值稳定性：减去最大值
    max_x = k.max(x, (-1,), True)
    x_sub = k.sub(x, max_x)
    
    # 计算exp
    exp_x = k.exp(x_sub)
    
    # 计算sum
    sum_exp = k.sum(exp_x, (-1,), True)
    
    # 归一化
    out = k.div(exp_x, sum_exp)
    
    return k.store(out)
```

### 8.3 GELU实现

```python
import math

@dvm.kernel
def gelu(k, x):
    x = k.load(x, dvm.float32)
    
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    
    x3 = k.mul(x, x)
    x3 = k.mul(x3, x)
    x3 = k.mul(x3, 0.044715)
    
    inner = k.add(x, x3)
    inner = k.mul(inner, sqrt_2_pi)
    
    # tanh近似实现
    exp_2x = k.exp(inner)
    exp_neg_2x = k.exp(k.mul(inner, -1.0))
    tanh_x = k.div(
        k.sub(exp_2x, exp_neg_2x),
        k.add(exp_2x, exp_neg_2x)
    )
    
    one_plus_tanh = k.add(1.0, tanh_x)
    out = k.mul(0.5, x)
    out = k.mul(out, one_plus_tanh)
    
    return k.store(out)
```

---

## 九、API速查表

### 9.1 Kernel类型常量

```python
dvm.Kernel.K_VEC     # Vector Kernel
dvm.Kernel.K_CUBE    # Cube Kernel
dvm.Kernel.K_MIX     # Mix Kernel
dvm.Kernel.K_PARAL   # Parallel Kernel
dvm.Kernel.K_SEQ     # Sequence Kernel
dvm.Kernel.K_SPLIT   # Split Kernel
dvm.Kernel.K_EAGER   # Eager Kernel
```

### 9.2 Kernel标志常量

```python
dvm.Kernel.F_DYN     # 动态Shape标志
dvm.Kernel.F_UWS     # 统一Workspace标志
dvm.Kernel.F_SPEC    # 投机执行标志
```

### 9.3 完整API列表

| 类别 | API | 说明 |
|-----|-----|------|
| 内存 | load, view_load, store | 数据加载存储 |
| 一元 | sqrt, abs, log, exp, reciprocal | 数学函数 |
| 一元 | round, floor, ceil, trunc | 取整函数 |
| 一元 | logical_not, isfinite | 逻辑函数 |
| 二元 | add, sub, mul, div, pow | 算术运算 |
| 二元 | equal, not_equal, greater, less | 比较运算 |
| 二元 | maximum, minimum | 极值运算 |
| 二元 | logical_and, logical_or | 逻辑运算 |
| 归约 | sum, max, min | Reduce操作 |
| 矩阵 | matmul, grouped_matmul | 矩阵乘法 |
| 形状 | reshape, broadcast, copy | 形状操作 |
| 类型 | cast | 类型转换 |
| 条件 | select | 条件选择 |
| 通信 | allreduce, allgather, reducescatter | 通信算子 |
| 调试 | dump, das | 调试输出 |

---

## 十、总结

本文详细介绍了DVM的Python接口架构和使用方法：

1. **核心类**：DataType、NDObject、IntArrayRef、ScalarRef
2. **Kernel API**：内存操作、计算操作、矩阵乘法等
3. **装饰器机制**：@dvm.kernel的工作原理
4. **测试工具**：Tester类的使用方法
5. **高级特性**：Parallel、Sequence、通信算子
6. **调试技巧**：Dump、反汇编、Verbose模式
7. **实战案例**：LayerNorm、Softmax、GELU实现

通过掌握这些内容，你可以高效地使用DVM开发自定义算子，实现高性能的算子融合优化。

---

## 参考资料

- [DVM官方仓库](https://gitcode.com/mindspore/dvm)
- [用户开发指南](../docs/tutorial.md)
- [Python接口定义](../python/dvm/_dvm_py.pyi)
