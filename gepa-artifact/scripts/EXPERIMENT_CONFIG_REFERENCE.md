# GEPA 实验配置参考

## 什么是 Program？

**Program** = 解决任务的具体策略/算法实现

- 一个 **Benchmark（基准测试）** 定义了"要解决什么问题"（数据集）
- 一个 **Program（程序）** 定义了"如何解决这个问题"（算法/提示词链）
- 同一个 Benchmark 可以有多个不同的 Program 实现

### 示例：HotpotMultiHop

```python
class HotpotMultiHop(dspy.Module):
    """3跳推理策略：问题 → 检索 → 总结 → 新查询 → 再检索 → 再总结 → 答案"""

    def forward(self, question):
        # 第1跳：初始检索
        hop1_docs = self.retrieve(question)
        summary_1 = self.summarize(question, hop1_docs)

        # 第2跳：基于总结生成新查询
        hop2_query = self.create_query(question, summary_1)
        hop2_docs = self.retrieve(hop2_query)
        summary_2 = self.summarize(question, summary_1, hop2_docs)

        # 第3跳：综合答案
        answer = self.final_answer(question, summary_1, summary_2)
        return answer
```

---

## 可用的 Benchmark 和 Program

### 1. HoVer（事实验证）
- **索引**: `bm_idx=0`
- **名称**: `HoverBench`
- **任务**: 基于多个维基百科文章验证陈述的真伪

| Program 索引 | Program 名称 | 策略描述 |
|-------------|-------------|----------|
| 0 | `HoverMultiHop` | 多跳检索验证策略 |

**配置示例**:
```bash
BENCHMARK_IDX=0
BENCHMARK_NAME="HoverBench"
PROGRAM_IDX=0
PROGRAM_NAME="HoverMultiHop"
```

---

### 2. HotpotQA（多跳问答）
- **索引**: `bm_idx=1`
- **名称**: `HotpotQABench`
- **任务**: 回答需要多步推理的复杂问题

| Program 索引 | Program 名称 | 策略描述 |
|-------------|-------------|----------|
| 0 | `HotpotMultiHop` | 3跳推理策略 |

**配置示例**:
```bash
BENCHMARK_IDX=1
BENCHMARK_NAME="HotpotQABench"
PROGRAM_IDX=0
PROGRAM_NAME="HotpotMultiHop"
```

---

### 3. Papillon（推理任务）
- **索引**: `bm_idx=2`
- **名称**: `Papillon`
- **任务**: 复杂推理问题

| Program 索引 | Program 名称 | 策略描述 |
|-------------|-------------|----------|
| 0 | `PAPILLON` | 标准推理策略 |

**配置示例**:
```bash
BENCHMARK_IDX=2
BENCHMARK_NAME="Papillon"
PROGRAM_IDX=0
PROGRAM_NAME="PAPILLON"
```

---

### 4. IFBench（指令跟随）
- **索引**: `bm_idx=3`
- **名称**: `IFBench`
- **任务**: 精确遵循用户指令

| Program 索引 | Program 名称 | 策略描述 |
|-------------|-------------|----------|
| 0 | `IFBenchCoT2StageProgram` | 两阶段思维链策略 |

**配置示例**:
```bash
BENCHMARK_IDX=3
BENCHMARK_NAME="IFBench"
PROGRAM_IDX=0
PROGRAM_NAME="IFBenchCoT2StageProgram"
```

---

### 5. LiveBench Math（数学问题）
- **索引**: `bm_idx=4`
- **名称**: `LiveBenchMathBench`
- **任务**: 解决数学问题

| Program 索引 | Program 名称 | 策略描述 |
|-------------|-------------|----------|
| 0 | `CoT` | 思维链推理 |

**配置示例**:
```bash
BENCHMARK_IDX=4
BENCHMARK_NAME="LiveBenchMathBench"
PROGRAM_IDX=0
PROGRAM_NAME="CoT"
```

---

### 6. AIME（数学竞赛）
- **索引**: `bm_idx=5`
- **名称**: `AIMEBench`
- **任务**: 美国数学邀请赛题目

| Program 索引 | Program 名称 | 策略描述 |
|-------------|-------------|----------|
| 0 | `CoT` | 思维链推理 |

**配置示例**:
```bash
BENCHMARK_IDX=5
BENCHMARK_NAME="AIMEBench"
PROGRAM_IDX=0
PROGRAM_NAME="CoT"
```

---

## 可用的优化器

| 优化器索引 | 优化器名称 | 描述 |
|-----------|-----------|------|
| 0 | `Baseline` | 无优化，直接运行 |
| 1 | `MIPROv2-Heavy` | 重度提示优化（自动优化提示词）|
| 2 | `GEPA-MERGE` | GEPA 合并版本（本文方法变体）|
| 3 | `GEPA` | 标准 GEPA（本文主要方法）|
| 4 | `Abl-SelectBestCandidate` | 消融研究：只选择最佳候选 |
| 5 | `GRPO` | 强化学习优化（需要更多GPU）|

---

## 可用的语言模型

### 本地模型（需要本地GPU）

```bash
LM_NAME="qwen3-8b"
LM_MODEL="openai/arbor:qwen/qwen3-8b"  # 或者绝对路径
# 示例: "openai/arbor:/home/yuhan/model_zoo/Qwen2.5-8B-Instruct"
```

**GPU需求**:
- Baseline/MIPROv2/GEPA: 4个GPU（推理模式）
- GRPO: 3个GPU（训练模式）

### 远程API模型

```bash
LM_NAME="gpt-41-mini"
LM_MODEL="openai/gpt-4.1-mini-2025-04-14"
# 需要设置: export OPENAI_API_KEY=sk-xxx
```

---

## 完整配置示例

### 示例1: 快速测试（Baseline）

```bash
# 基准测试
BENCHMARK_IDX=1
BENCHMARK_NAME="HotpotQABench"

# 程序（解决策略）
PROGRAM_IDX=0
PROGRAM_NAME="HotpotMultiHop"

# 优化器
OPTIMIZER_IDX=0
OPTIMIZER_NAME="Baseline"

# 语言模型
LM_NAME="qwen3-8b"

# 其他
SEED=0
NUM_THREADS=32
DRY_RUN=false
```

### 示例2: 完整GEPA实验

```bash
# 基准测试
BENCHMARK_IDX=4
BENCHMARK_NAME="LiveBenchMathBench"

# 程序
PROGRAM_IDX=0
PROGRAM_NAME="CoT"

# 优化器
OPTIMIZER_IDX=3
OPTIMIZER_NAME="GEPA"

# 语言模型
LM_NAME="qwen3-8b"

# 其他
SEED=0
NUM_THREADS=32
USE_CACHE_FROM_OPT="MIPROv2-Heavy"  # 使用MIPROv2的缓存
```

---

## 注意事项

1. **Program 通常只有索引 0**
   - 目前每个 Benchmark 只实现了一个 Program
   - 但框架支持同一任务的多种解决策略

2. **名称必须完全匹配**
   - `BENCHMARK_NAME` 和 `PROGRAM_NAME` 必须与代码中的类名完全一致
   - 大小写敏感！

3. **缓存依赖**
   - GEPA 依赖 MIPROv2-Heavy 的缓存
   - 运行 GEPA 前需要先运行 MIPROv2-Heavy

4. **GPU 要求**
   - 不同优化器对GPU数量有不同要求
   - 查看 `experiment_configs.py` 中的 `launch_arbor` 配置