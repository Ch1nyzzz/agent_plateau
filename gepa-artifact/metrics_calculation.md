# GEPA 实验指标计算详解

## 概述

在运行 GEPA 实验时，系统会在 `evaluation_results/evaluation_result.txt` 文件中记录多项性能和成本指标。本文档详细说明这些指标的计算方式。

## 输出文件位置

实验结果保存在以下路径：
```
experiment_runs_data/experiment_runs/seed_{seed}/{benchmark}_{program}_{optimizer}_{model}/evaluation_results/evaluation_result.txt
```

例如：
```
experiment_runs_data/experiment_runs/seed_0/HotpotQABench_HotpotMultiHop_GEPA-5_gpt-41-mini/evaluation_results/evaluation_result.txt
```

## 输出文件格式

文件为 CSV 格式，包含以下列：

```csv
score,cost,input_tokens,output_tokens,optimizer,optimizer_cost,optimizer_input_tokens,optimizer_output_tokens
```

示例数据行：
```csv
60.0,0.8387668,1277283,204414,GEPA-5,3.7305671999999968,2427346,318402,
```

## 指标详解

### 1. 评估阶段指标（Evaluation Metrics）

这些指标反映最终模型在测试集上的表现：

#### 1.1 score（评估分数）
- **含义**: 模型在测试集上的平均性能得分
- **计算位置**: `scripts/run_experiments.py:444`
- **计算方式**:
  ```python
  score = evaluate_prog(optimized_program)
  ```
  使用 `dspy.Evaluate` 评估优化后的程序在测试集上的表现

#### 1.2 cost（评估成本）
- **含义**: 在测试集评估阶段调用语言模型的总成本（美元）
- **计算位置**: `scripts/run_experiments.py:446-448`
- **计算方式**: 通过 `calculate_stats()` 函数累加所有 API 调用的成本
  ```python
  cost, input_tokens, output_tokens = calculate_stats(eval_lm)
  ```

#### 1.3 input_tokens（评估输入 token 数）
- **含义**: 在测试集评估阶段，发送给语言模型的所有输入 token 总数
- **计算位置**: `scripts/run_experiments.py:446-448`
- **计算方式**: 累加所有 API 调用的 `prompt_tokens`

#### 1.4 output_tokens（评估输出 token 数）
- **含义**: 在测试集评估阶段，语言模型生成的所有输出 token 总数
- **计算位置**: `scripts/run_experiments.py:446-448`
- **计算方式**: 累加所有 API 调用的 `completion_tokens`

### 2. 优化阶段指标（Optimization Metrics）

这些指标反映在训练/验证集上进行优化的成本（仅当使用优化器时才有值）：

#### 2.1 optimizer（优化器名称）
- **含义**: 使用的优化器名称（如 GEPA-5、GEPA-10 等）
- **数据来源**: 命令行参数 `--optim_name`
- **对于 Baseline**: 此字段为空

#### 2.2 optimizer_cost（优化成本）
- **含义**: 在优化阶段（训练/验证集）调用语言模型的总成本（美元）
- **计算位置**: `scripts/run_experiments.py:422-425`
- **计算方式**:
  ```python
  optimizer_cost, optimizer_input_tokens, optimizer_output_tokens = calculate_stats(lm_for_optimizer)
  ```
  在优化器编译完成后计算

#### 2.3 optimizer_input_tokens（优化输入 token 数）
- **含义**: 在优化阶段，发送给语言模型的所有输入 token 总数
- **计算位置**: `scripts/run_experiments.py:422-425`
- **计算方式**: 累加优化阶段所有 API 调用的 `prompt_tokens`

#### 2.4 optimizer_output_tokens（优化输出 token 数）
- **含义**: 在优化阶段，语言模型生成的所有输出 token 总数
- **计算位置**: `scripts/run_experiments.py:422-425`
- **计算方式**: 累加优化阶段所有 API 调用的 `completion_tokens`

## 核心计算函数

### calculate_stats() 函数

位置：`scripts/run_experiments.py:47-56`

```python
def calculate_stats(lm) -> tuple[float, int, int]:
    cost = 0
    input_tokens = 0
    output_tokens = 0
    for i, trace in enumerate(lm.history):
        cost += trace.get("cost", None) or 0
        input_tokens += trace.get("usage", 0).get("prompt_tokens", 0)
        output_tokens += trace.get("usage", 0).get("completion_tokens", 0)

    return cost, input_tokens, output_tokens
```

**工作原理**：
1. 遍历语言模型对象（`lm`）的历史调用记录（`lm.history`）
2. 对于每次 API 调用的 trace：
   - 累加 `cost` 字段（API 调用成本）
   - 累加 `usage.prompt_tokens`（输入 token 数）
   - 累加 `usage.completion_tokens`（输出 token 数）
3. 返回三个累加值

**关键点**：
- `lm.history` 由 DSPy 框架自动维护，记录所有 API 调用
- 每次 API 调用后，OpenAI 等提供商会返回 token 使用情况
- 成本根据模型定价和 token 使用量自动计算

### write_evaluation_result_to_path() 函数

位置：`scripts/run_experiments.py:18-45`

```python
def write_evaluation_result_to_path(evaluation_result, file_path):
    os.makedirs(file_path, exist_ok=True)
    file_name = f"evaluation_result"
    if evaluation_result.optimizer:
        optimizer_header = "optimizer,optimizer_cost,optimizer_input_tokens,optimizer_output_tokens"
        optimizer_values = (
            f"{evaluation_result.optimizer},{evaluation_result.optimizer_cost},"
            f"{evaluation_result.optimizer_input_tokens},{evaluation_result.optimizer_output_tokens},"
        )
    else:
        optimizer_header = ""
        optimizer_values = ""
    with open(os.path.join(file_path, f"{file_name}.txt"), "w") as f:
        f.write(f"score,cost,input_tokens,output_tokens,{optimizer_header}\n")
        f.write(
            f"{evaluation_result.score},{evaluation_result.cost},{evaluation_result.input_tokens},"
            f"{evaluation_result.output_tokens},{optimizer_values}\n"
        )
```

**工作原理**：
1. 创建 `evaluation_results` 目录
2. 根据是否使用优化器决定 CSV 列
3. 写入 CSV 格式的结果文件

## 数据结构

### EvaluationResult 类

位置：`gepa_artifact/benchmarks/benchmark.py:88-103`

```python
@dataclass
class EvaluationResult:
    benchmark: str
    program: str

    score: float = None
    cost: float = None
    input_tokens: int = None
    output_tokens: int = None

    optimizer: str = None
    optimized_program: dspy.Module = None
    optimizer_input_tokens: int = None
    optimizer_output_tokens: int = None
    optimizer_cost: float = None

    optimizer_program_scores: list[float] = None
```

这个数据类存储了所有评估结果，包括评估阶段和优化阶段的指标。

## 实验流程

### 完整流程图

```
1. 运行 run_batch_experiments.py
   ↓
2. 调用 scripts.run_experiments 模块
   ↓
3. [如果有优化器] 执行优化阶段
   - 创建 lm_for_optimizer
   - 运行 optimizer.compile()
   - 调用 calculate_stats(lm_for_optimizer)
   - 记录 optimizer_cost, optimizer_input_tokens, optimizer_output_tokens
   ↓
4. 执行评估阶段
   - 创建 eval_lm
   - 在测试集上评估 optimized_program
   - 调用 calculate_stats(eval_lm)
   - 记录 score, cost, input_tokens, output_tokens
   ↓
5. 调用 write_evaluation_result_to_path()
   - 将所有指标写入 evaluation_result.txt
```

### 代码关键路径

```python
# 1. 优化阶段 (scripts/run_experiments.py:354-431)
if optim_name != "Baseline":
    optimizer = optimizer_config.optimizer(metric=metric_fn_with_logger, **init_args)
    lm_for_optimizer = create_lm(lm_config)
    dspy.configure(lm=lm_for_optimizer)

    # 执行优化
    optimized_program = optimizer.compile(program, trainset=..., valset=...)

    # 计算优化阶段指标
    (optimizer_cost, optimizer_input_tokens, optimizer_output_tokens) = calculate_stats(lm_for_optimizer)
    eval_results.optimizer = optim_name
    eval_results.optimizer_cost = optimizer_cost
    eval_results.optimizer_input_tokens = optimizer_input_tokens
    eval_results.optimizer_output_tokens = optimizer_output_tokens

# 2. 评估阶段 (scripts/run_experiments.py:433-451)
evaluate_prog = dspy.Evaluate(devset=final_eval_set, metric=metric_fn_with_logger, ...)
eval_lm = create_lm(lm_config)
dspy.configure(lm=eval_lm)

# 在测试集上评估
score = evaluate_prog(optimized_program)
eval_results.score = score

# 计算评估阶段指标
cost, input_tokens, output_tokens = calculate_stats(eval_lm)
eval_results.cost = cost
eval_results.input_tokens = input_tokens
eval_results.output_tokens = output_tokens

# 3. 写入文件 (scripts/run_experiments.py:453-456)
write_evaluation_result_to_path(eval_results, os.path.join(runs_dir, "evaluation_results"))
```

## Token 计数来源

Token 使用统计来自 OpenAI API（或其他 LLM 提供商）的响应。每次 API 调用都会返回类似以下的使用信息：

```json
{
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801
  },
  "cost": 0.012345
}
```

DSPy 框架会自动捕获这些信息并存储在 `lm.history` 中，供后续统计使用。

## 成本计算

成本计算基于：
1. **输入 token 成本**: `input_tokens × input_price_per_token`
2. **输出 token 成本**: `output_tokens × output_price_per_token`
3. **总成本**: 输入成本 + 输出成本

不同模型有不同的定价，例如 GPT-4.1-mini：
- 输入: $0.0015 / 1K tokens
- 输出: $0.006 / 1K tokens

## 注意事项

1. **优化器指标仅在非 Baseline 实验中存在**
   - Baseline 实验不进行优化，因此 optimizer 相关字段为空

2. **Token 统计的准确性**
   - 依赖于 LLM 提供商返回的准确信息
   - DSPy 框架会自动处理重试和错误情况

3. **成本累加**
   - 包括所有 API 调用，包括失败后的重试
   - 包括优化过程中的所有评估调用

4. **缓存机制**
   - 使用 `--use_cache_from_opt` 可以重用 Baseline 的缓存
   - 这可以减少重复的 API 调用和成本

## 实际示例解读

以这个实际结果为例：
```csv
score,cost,input_tokens,output_tokens,optimizer,optimizer_cost,optimizer_input_tokens,optimizer_output_tokens
60.0,0.8387668,1277283,204414,GEPA-5,3.7305671999999968,2427346,318402,
```

**解读**：
- **score = 60.0**: 模型在测试集上获得 60% 的准确率
- **cost = 0.8387668**: 评估阶段花费约 0.84 美元
- **input_tokens = 1277283**: 评估阶段输入了约 127 万个 token
- **output_tokens = 204414**: 评估阶段生成了约 20 万个 token
- **optimizer = GEPA-5**: 使用 GEPA-5 优化器
- **optimizer_cost = 3.7305672**: 优化阶段花费约 3.73 美元
- **optimizer_input_tokens = 2427346**: 优化阶段输入了约 242 万个 token
- **optimizer_output_tokens = 318402**: 优化阶段生成了约 31 万个 token

**总成本**: 0.84 + 3.73 = 4.57 美元

## 相关文件

- **主实验脚本**: `/home/yuhan/ReAct_learning/agent_plateau/gepa-artifact/run_batch_experiments.py`
- **实验执行逻辑**: `/home/yuhan/ReAct_learning/agent_plateau/gepa-artifact/scripts/run_experiments.py`
- **数据类定义**: `/home/yuhan/ReAct_learning/agent_plateau/gepa-artifact/gepa_artifact/benchmarks/benchmark.py`

## 总结

GEPA 实验的指标计算是一个两阶段的过程：

1. **优化阶段**（仅非 Baseline）：在训练/验证集上优化程序，记录优化成本
2. **评估阶段**：在测试集上评估优化后的程序，记录评估成本和性能

所有指标都通过 `calculate_stats()` 函数从 DSPy 的 LM 历史记录中自动提取和累加，确保了统计的准确性和完整性。
