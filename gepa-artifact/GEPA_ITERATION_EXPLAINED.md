# GEPA 迭代优化过程详解

## 什么是 Iteration（迭代）？

在 GEPA 中，**每一次 iteration 代表一次 prompt 优化尝试**。每个迭代都会：
1. 选择一个程序候选
2. 选择其中一个 predictor（模块）
3. 为该 predictor 生成新的 instruction
4. 评估新 instruction 的效果
5. 如果更好，就添加到候选池中

## 迭代过程详细流程

```
迭代 N：
┌─────────────────────────────────────────────────────────────┐
│ 1. 选择阶段                                                  │
│    ├─ 从候选程序池中选择一个程序 (Program Candidate)         │
│    └─ 选择该程序中的一个 Predictor 进行优化                  │
├─────────────────────────────────────────────────────────────┤
│ 2. 采样阶段                                                  │
│    └─ 从训练集中采样一个 mini-batch（subsample）             │
├─────────────────────────────────────────────────────────────┤
│ 3. 反馈收集阶段                                              │
│    ├─ 在 subsample 上运行当前程序                            │
│    ├─ 收集错误样本的反馈（哪里做错了）                        │
│    └─ 计算当前的 subsample_score                             │
├─────────────────────────────────────────────────────────────┤
│ 4. Instruction 生成阶段                                      │
│    ├─ 将反馈和错误样本发送给 teacher LM                       │
│    ├─ Teacher LM 分析错误原因                                │
│    └─ Teacher LM 生成新的 instruction                        │
├─────────────────────────────────────────────────────────────┤
│ 5. 评估阶段                                                  │
│    ├─ 创建带有新 instruction 的程序副本                       │
│    ├─ 在 subsample 上评估新程序                              │
│    └─ 计算 new_subsample_score                               │
├─────────────────────────────────────────────────────────────┤
│ 6. 决策阶段                                                  │
│    ├─ 如果 new_subsample_score > subsample_score:           │
│    │   ├─ 在完整数据集上评估新程序                           │
│    │   ├─ 将新程序添加到候选池                               │
│    │   └─ 更新 Pareto Front                                 │
│    └─ 否则：丢弃新程序，进入下一次迭代                        │
├─────────────────────────────────────────────────────────────┤
│ 7. 合并阶段（可选，如果启用了 merge）                         │
│    ├─ 如果上一次迭代找到了新程序                             │
│    ├─ 尝试合并 Pareto Front 上的优秀程序                     │
│    └─ 评估合并后的程序                                       │
└─────────────────────────────────────────────────────────────┘
```

## 关键概念

### 1. Program Candidate（程序候选）
- 代表一个完整的 prompt 程序（包含多个 predictors）
- GEPA 维护一个候选程序池，不断添加新的改进版本

### 2. Predictor（预测器）
- 程序中的一个模块，负责特定的子任务
- 每个 predictor 都有自己的 instruction
- 例如：在多跳问答中可能有 `hop1`、`hop2`、`generate_answer` 等

### 3. Subsample（子采样）
- 每次迭代只在一小部分训练数据上优化
- 默认大小：`minibatch_size=25` 个样本
- 这样可以快速迭代，不用每次都跑完整数据集

### 4. Pareto Front（帕累托前沿）
- 保存所有"非劣解"的程序
- 如果一个程序在某些样本上表现更好，即使总分不是最高，也会被保留
- 维护多样性，避免过早收敛

### 5. Full Evaluation（完整评估）
- 只有当新程序在 subsample 上表现更好时，才会在完整数据集上评估
- 节省计算资源

## 迭代终止条件

GEPA 会在满足以下任一条件时停止：

```python
while (
    (num_iters is None or num_full_ds_evals < num_iters) and
    (max_evals_per_trainval_instance is None or total_num_evals_per_trainval_instance < max_evals_per_trainval_instance) and
    (max_metric_calls is None or total_num_evals < max_metric_calls)
):
```

1. **`num_iters`**: 完整数据集评估次数达到上限
2. **`max_evals_per_trainval_instance`**: 平均每个样本被评估的次数达到上限
3. **`max_metric_calls`**: 总的 metric 调用次数达到上限

## 优化策略

### 1. **渐进式优化**
```
迭代1: 优化 predictor_0 -> 程序 v1
迭代2: 优化 predictor_1 -> 程序 v2
迭代3: 优化 predictor_2 -> 程序 v3
迭代4: 优化 predictor_0（基于 v1）-> 程序 v4
...
```
- 轮流优化每个 predictor
- 每次只改变一个 instruction
- 类似坐标下降法

### 2. **Evolutionary Search（进化搜索）**
```
候选池:
  程序 0: [inst_A, inst_B, inst_C] - score: 70
  程序 1: [inst_A', inst_B, inst_C] - score: 75
  程序 2: [inst_A, inst_B', inst_C] - score: 72
  程序 3: [inst_A', inst_B', inst_C] - score: 80 ← 最好
```
- 维护多个程序候选
- 优先选择 Pareto Front 上的程序进行优化
- 保持多样性，避免局部最优

### 3. **Feedback-Driven（反馈驱动）**
```
错误样本 → 分析原因 → 生成针对性的 instruction
```
- 不是盲目搜索，而是基于具体错误生成改进建议
- Teacher LM 分析失败案例，提出改进方向

### 4. **Program Merging（程序合并）**
```
程序 A: [inst_A, inst_B_old, inst_C]
程序 B: [inst_A_old, inst_B', inst_C]
         ↓ 合并
新程序: [inst_A, inst_B', inst_C]
```
- 将不同程序中表现好的 instructions 组合起来
- 探索组合空间，可能产生更好的解

## 计算成本分析

### 低成本操作（每次迭代）
- Subsample 评估：25 个样本
- Instruction 生成：1 次 LM 调用

### 高成本操作（只在改进时）
- 完整数据集评估：所有样本
- 只有当 `new_subsample_score > subsample_score` 时才执行

### 示例计算
假设：
- 训练集大小：300 个样本
- Minibatch 大小：25 个样本
- 运行 50 次迭代
- 其中 10 次找到了改进

总评估次数：
- Subsample 评估：50 × 25 = 1,250 次
- 完整评估：10 × 300 = 3,000 次
- **总计：4,250 次**

如果每次都完整评估：50 × 300 = 15,000 次
**节省了约 72% 的计算量**

## 实际例子

假设优化一个 2-hop QA 任务：

```
初始程序:
  hop1: "Generate a search query."
  hop2: "Generate another search query."
  answer: "Answer the question."

迭代1: 优化 hop1
  → 发现：查询不够具体
  → 新 instruction: "Generate a specific search query focusing on the key entities."
  → 评估：改进！添加到候选池

迭代2: 优化 hop2
  → 发现：没有利用第一跳的信息
  → 新 instruction: "Based on hop1 results, generate a follow-up query."
  → 评估：改进！添加到候选池

迭代3: 优化 answer
  → 发现：答案不够简洁
  → 新 instruction: "Provide a concise answer based on all search results."
  → 评估：改进！添加到候选池

...继续迭代直到收敛
```

最终可能得到多个候选程序，用户可以选择在验证集上表现最好的。

## 查看迭代日志

在您的输出中可以看到：
```
2025/11/02 01:40:58 INFO dspy.evaluate.evaluate: Average Metric: 133 / 300 (44.3%)
```

这表示某次评估的结果。完整的迭代日志会保存在：
- `experiment_runs_data/experiment_runs/.../logs/`
- WandB 上也能看到每次迭代的详细指标

## 总结

**Iteration（迭代）= 一次完整的"发现错误 → 生成新 instruction → 评估改进"循环**

GEPA 通过：
1. ✅ **高效采样**：只在 mini-batch 上快速试错
2. ✅ **反馈驱动**：基于具体错误生成改进
3. ✅ **进化搜索**：维护多个候选，保持多样性
4. ✅ **渐进优化**：每次只改一个模块
5. ✅ **程序合并**：组合优秀的 instructions

来实现自动化的 prompt 优化！
