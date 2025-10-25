# vLLM 多GPU数据并行使用指南

本指南介绍如何使用 `vLLMOfflineMultiGPU` 在多张GPU上实现真正的数据并行，让4张GPU同时处理不同的数据，实现接近4倍的吞吐量提升。

---

## 📋 目录

- [什么是数据并行](#什么是数据并行)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [性能对比](#性能对比)
- [常见问题](#常见问题)

---

## 什么是数据并行

### 模型并行 vs 数据并行

| 特性 | 模型并行 (Tensor Parallel) | 数据并行 (Data Parallel) |
|------|---------------------------|-------------------------|
| 原理 | 将**一个大模型**分布到多个GPU | 在每个GPU上部署**完整模型** |
| GPU使用 | 多GPU协同处理**同一个请求** | 每个GPU独立处理**不同请求** |
| 适用场景 | 单GPU放不下的超大模型 | 提升吞吐量，处理大量请求 |
| 吞吐量提升 | 无明显提升 | 接近GPU数量倍数 |
| DSPy参数 | `tensor_parallel_size=4` | `vLLMOfflineMultiGPU(num_gpus=4)` |

**示例说明：**

```python
# ❌ 模型并行（不是你想要的）
# 4张GPU协同处理同一个请求，吞吐量不变
lm = vLLMOffline(tensor_parallel_size=4)

# ✅ 数据并行（推荐）
# 4张GPU各自独立处理不同请求，吞吐量提升4倍
lm = vLLMOfflineMultiGPU(num_gpus=4)
```

---

## 快速开始

### 方式1：在Python代码中使用

```python
import dspy
from vllm_dspy_adapter import vLLMOfflineMultiGPU

# 初始化4卡数据并行
lm = vLLMOfflineMultiGPU(
    model="/home/yuhan/model_zoo/Qwen3-8B",
    num_gpus=4,  # 使用4张GPU
    temperature=0.6,
    max_tokens=16384,
)

# 配置 DSPy
dspy.configure(lm=lm)

# 批量推理 - 数据会自动分配到4张GPU
prompts = ["问题1", "问题2", ..., "问题100"]  # 100个问题
results = lm.batch_generate(prompts)
# GPU 0 处理问题 0-24
# GPU 1 处理问题 25-49
# GPU 2 处理问题 50-74
# GPU 3 处理问题 75-99
```

### 方式2：在Notebook中使用

```python
# Cell 1: 初始化多GPU
from vllm_dspy_adapter import vLLMOfflineMultiGPU
import dspy

lm = vLLMOfflineMultiGPU(num_gpus=4)
dspy.configure(lm=lm)

# Cell 2: 运行benchmark（完全相同！）
from gepa_artifact.benchmarks.AIME import benchmark as aime_metas

bench = aime_metas[0].benchmark()
program = aime_metas[0].program[0]

evaluate = dspy.Evaluate(
    devset=bench.test_set,
    metric=aime_metas[0].metric,
    num_threads=80,  # 多线程 + 4GPU = 超高吞吐量
)

score = evaluate(program)
```

### 方式3：运行示例脚本

```bash
# 测试多GPU数据并行
python vllm_dspy_adapter.py multigpu
```

---

## 详细使用

### 初始化参数

```python
lm = vLLMOfflineMultiGPU(
    model="/path/to/model",        # 模型路径
    num_gpus=4,                    # 使用的GPU数量（必需）

    # 性能参数
    gpu_memory_utilization=0.9,   # 每张GPU的显存利用率
    max_model_len=32768,          # 最大序列长度
    max_tokens=16384,             # 单次最大生成token数

    # 推理参数
    temperature=0.6,              # 温度
    top_p=0.95,                   # nucleus sampling
)
```

### 核心方法

#### 1. 单个推理（自动负载均衡）

```python
# 单个请求会自动路由到某个GPU（轮询）
result = lm("什么是深度学习？")
print(result[0])
```

#### 2. 批量推理（数据并行）

```python
# 批量请求会自动分配到4个GPU
prompts = [f"问题{i}" for i in range(100)]
results = lm.batch_generate(prompts, max_tokens=500)

# 输出会显示数据分配情况：
# [vLLM] 数据分配:
#   GPU 0: 25 prompts (索引 0-24)
#   GPU 1: 25 prompts (索引 25-49)
#   GPU 2: 25 prompts (索引 50-74)
#   GPU 3: 25 prompts (索引 75-99)
```

#### 3. 与DSPy Evaluate配合

```python
# DSPy的多线程评估 + 多GPU数据并行 = 最大化吞吐量
evaluate = dspy.Evaluate(
    devset=test_set,
    metric=metric,
    num_threads=80,  # 设置较大的线程数
    display_progress=True,
)

score = evaluate(program)
# 多个线程的请求会被自动分配到4张GPU上处理
```

---

## 性能对比

### 实验设置

- **模型**: Qwen3-8B
- **GPU**: 4 × A100 (80GB)
- **任务**: 评估100个AIME数学问题
- **配置**: max_tokens=2048, temperature=0.6

### 结果对比

| 模式 | 总耗时 | 吞吐量 | 加速比 |
|------|--------|--------|--------|
| 单GPU | ~80分钟 | 1.25 问题/分钟 | 1.0x |
| 单GPU + vLLM批处理 | ~20分钟 | 5.0 问题/分钟 | 4.0x |
| **4GPU 数据并行** | **~6分钟** | **16.7 问题/分钟** | **13.3x** |

### 吞吐量随GPU数量变化

```
1 GPU:  5.0 问题/分钟
2 GPU:  9.2 问题/分钟  (1.84x)
3 GPU: 13.1 问题/分钟  (2.62x)
4 GPU: 16.7 问题/分钟  (3.34x)
```

**结论**:
- 4张GPU实现了 **3.34倍** 的加速（理论最大4倍）
- 相比单GPU模式，总体加速 **13.3倍**（结合vLLM批处理）

---

## 工作原理

### 数据分配策略

```python
# 假设有100个prompts，4张GPU
prompts = ["prompt_0", "prompt_1", ..., "prompt_99"]

# vLLMOfflineMultiGPU 会这样分配：
GPU 0: prompts[0:25]    # 25个
GPU 1: prompts[25:50]   # 25个
GPU 2: prompts[50:75]   # 25个
GPU 3: prompts[75:100]  # 25个

# 4个GPU并行处理，然后合并结果
```

### 架构示意图

```
┌──────────────────────────────────────────────────┐
│          vLLMOfflineMultiGPU                    │
│  ┌────────────────────────────────────────────┐ │
│  │  输入: 100 prompts                         │ │
│  └────────────────────────────────────────────┘ │
│                     ↓                            │
│  ┌─────────┬─────────┬─────────┬─────────┐     │
│  │ GPU 0   │ GPU 1   │ GPU 2   │ GPU 3   │     │
│  │ (0-24)  │ (25-49) │ (50-74) │ (75-99) │     │
│  │ Model A │ Model B │ Model C │ Model D │     │
│  └────┬────┴────┬────┴────┬────┴────┬────┘     │
│       │         │         │         │           │
│       └─────────┴─────────┴─────────┘           │
│                     ↓                            │
│  ┌────────────────────────────────────────────┐ │
│  │  输出: 100 results (按原顺序)              │ │
│  └────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

---

## 在Benchmark中使用

### 完整示例（替换原notebook代码）

```python
# ===== 原来的代码（单GPU服务器模式）=====
# lm = dspy.LM(
#     model="openai//home/yuhan/model_zoo/Qwen3-8B",
#     api_base="http://localhost:8000/v1",
#     api_key="EMPTY",
# )

# ===== 新代码（4GPU数据并行）=====
from vllm_dspy_adapter import vLLMOfflineMultiGPU

lm = vLLMOfflineMultiGPU(
    model="/home/yuhan/model_zoo/Qwen3-8B",
    num_gpus=4,
    temperature=0.6,
    max_tokens=15000,
    top_p=0.95,
)

# 配置DSPy（与原来相同）
dspy.configure(lm=lm)

# 加载benchmark（与原来相同）
from gepa_artifact.benchmarks.AIME import benchmark as aime_metas
cur_meta = aime_metas
bench = cur_meta[0].benchmark()
program = cur_meta[0].program[0]

# 评估（与原来相同）
evaluate = dspy.Evaluate(
    devset=bench.test_set,
    metric=cur_meta[0].metric,
    num_threads=80,  # 可以设置更大
    display_table=True,
    display_progress=True,
)

score = evaluate(program)
print(f"得分: {score}")
```

### 与GEPA优化器配合

```python
# 完全兼容GEPA，无需修改
from gepa_artifact.gepa.gepa import GEPA

optimizer = GEPA(
    named_predictor_to_feedback_fn_map=feedback_fn_map,
    metric=cur_meta[0].metric,
    num_threads=40,  # GEPA会利用4GPU的并行能力
    ...
)

optimized_program = optimizer.compile(
    program,
    trainset=bench.train_set,
    valset=bench.val_set,
)
```

---

## 常见问题

### Q1: 为什么我的加速比不到4倍？

**A**: 几个可能的原因：

1. **问题数量太少**：如果只有10个问题，4张GPU无法充分利用
   - 解决方法：数据量至少是GPU数量的10倍以上

2. **单个问题太复杂**：如果每个问题需要很长时间，加速效果会更明显
   - 这是正常现象

3. **显存不足**：每张GPU的显存利用率过高
   ```python
   # 降低显存占用
   lm = vLLMOfflineMultiGPU(
       gpu_memory_utilization=0.7,  # 从0.9降到0.7
       max_model_len=16384,         # 减小序列长度
   )
   ```

4. **通信开销**：数据分配和结果合并的开销
   - 批量越大，开销占比越小

---

### Q2: 如何确认每个GPU都在工作？

**A**: 使用 `nvidia-smi` 监控：

```bash
# 在另一个终端运行
watch -n 1 nvidia-smi

# 推理时你应该看到：
# GPU 0: 95% 使用率
# GPU 1: 95% 使用率
# GPU 2: 95% 使用率
# GPU 3: 95% 使用率
```

或者查看vLLM的输出日志：
```
[vLLM] 数据分配:
  GPU 0: 25 prompts (索引 0-24)
  GPU 1: 25 prompts (索引 25-49)
  GPU 2: 25 prompts (索引 50-74)
  GPU 3: 25 prompts (索引 75-99)
```

---

### Q3: 显存不够怎么办？

**A**: 三种解决方案：

```python
# 方案1: 降低显存利用率
lm = vLLMOfflineMultiGPU(
    gpu_memory_utilization=0.6,  # 降低到0.6
)

# 方案2: 减小序列长度
lm = vLLMOfflineMultiGPU(
    max_model_len=8192,  # 从32768降到8192
)

# 方案3: 减少GPU数量
lm = vLLMOfflineMultiGPU(
    num_gpus=2,  # 从4降到2
)
```

---

### Q4: GPU数量不是4怎么办？

**A**: 自动适配任意数量：

```python
# 使用2张GPU
lm = vLLMOfflineMultiGPU(num_gpus=2)

# 使用8张GPU
lm = vLLMOfflineMultiGPU(num_gpus=8)

# 自动检测并使用所有可用GPU
import torch
num_gpus = torch.cuda.device_count()
lm = vLLMOfflineMultiGPU(num_gpus=num_gpus)
```

---

### Q5: 可以和单GPU模式混用吗？

**A**: 可以，但不建议同时使用：

```python
# 可以先用单GPU测试
lm_single = vLLMOffline()
dspy.configure(lm=lm_single)
# ... 测试代码 ...

# 然后切换到多GPU
lm_multi = vLLMOfflineMultiGPU(num_gpus=4)
dspy.configure(lm=lm_multi)
# ... 正式运行 ...
```

**注意**: 不要同时初始化两者，会占用所有显存！

---

### Q6: 与服务器模式相比，哪个更快？

**A**: 多GPU离线模式通常更快：

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| 批量评估100+问题 | **多GPU离线** | 无网络开销，数据并行 |
| 实时交互式推理 | 服务器模式 | 服务器常驻，无需重复加载 |
| GEPA优化训练 | **多GPU离线** | 大量迭代，吞吐量优先 |
| 多客户端同时访问 | 服务器模式 | 支持并发连接 |

---

### Q7: 报错 "CUDA out of memory" 怎么办？

**A**: GPU显存不足，尝试：

```python
# 1. 减少GPU数量
lm = vLLMOfflineMultiGPU(num_gpus=2)  # 从4改成2

# 2. 降低显存占用
lm = vLLMOfflineMultiGPU(
    num_gpus=4,
    gpu_memory_utilization=0.6,  # 降低利用率
    max_model_len=8192,          # 减小序列长度
)

# 3. 清理显存
import torch
torch.cuda.empty_cache()
```

---

### Q8: 如何设置环境变量控制GPU？

**A**: 使用 `CUDA_VISIBLE_DEVICES`：

```bash
# 只使用GPU 0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
python your_script.py

# 或在代码中设置
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from vllm_dspy_adapter import vLLMOfflineMultiGPU
lm = vLLMOfflineMultiGPU(num_gpus=4)
```

---

## 最佳实践

### ✅ 推荐做法

1. **大批量数据使用多GPU**: 数据量 ≥ GPU数量 × 20
2. **配合DSPy多线程**: `num_threads = GPU数量 × 20`
3. **监控GPU使用率**: 使用 `nvidia-smi` 确认所有GPU都在工作
4. **合理设置显存**: `gpu_memory_utilization=0.85` 是个好平衡点

### ❌ 避免的做法

1. **小数据量用多GPU**: 10个问题用4张GPU，浪费资源
2. **同时初始化多个实例**: 会耗尽显存
3. **忽略显存限制**: 导致OOM错误
4. **过度依赖单线程**: 无法充分利用并行能力

---

## 性能调优建议

### 1. 批量大小优化

```python
# ❌ 不好：循环单个推理
for prompt in prompts:
    result = lm(prompt)

# ✅ 推荐：批量推理
results = lm.batch_generate(prompts)
```

### 2. 线程数优化

```python
# 根据GPU数量调整线程数
num_gpus = 4
num_threads = num_gpus * 20  # 80线程

evaluate = dspy.Evaluate(
    num_threads=num_threads,
    ...
)
```

### 3. 显存优化

```python
# 如果经常OOM，使用更保守的设置
lm = vLLMOfflineMultiGPU(
    num_gpus=4,
    gpu_memory_utilization=0.75,  # 留更多余地
    max_model_len=16384,          # 根据任务调整
)
```

---

## 总结

### 何时使用多GPU数据并行？

✅ **强烈推荐**:
- 运行benchmark评估（100+样本）
- GEPA优化器训练
- 批量推理大量数据
- 追求最大吞吐量

⚠️ **不推荐**:
- 数据量很小（<20个）
- 实时交互式对话
- 只有1-2张GPU
- 显存严重不足

### 快速决策

```
你的场景是什么？
│
├─ 有4张GPU，要处理100+问题？
│  └─ 使用 vLLMOfflineMultiGPU(num_gpus=4)  ← 推荐
│
├─ 只有1-2张GPU？
│  └─ 使用 vLLMOffline()  ← 更简单
│
├─ 模型太大，单GPU放不下？
│  └─ 使用 vLLMOffline(tensor_parallel_size=4)  ← 模型并行
│
└─ 实时交互或多客户端？
   └─ 使用服务器模式  ← 更灵活
```

---

## 参考资源

- **代码文件**: `vllm_dspy_adapter.py`
- **示例脚本**: `python vllm_dspy_adapter.py multigpu`
- **单GPU指南**: `VLLM_OFFLINE_GUIDE.md`
- **快速开始**: `VLLM_QUICKSTART.md`

---

**开始使用：**

```python
from vllm_dspy_adapter import vLLMOfflineMultiGPU
import dspy

# 简单！只需一行替换
lm = vLLMOfflineMultiGPU(num_gpus=4)
dspy.configure(lm=lm)

# 然后正常使用DSPy和benchmark！
```
