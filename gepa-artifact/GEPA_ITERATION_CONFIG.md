# GEPA 迭代参数配置说明

## 核心迭代参数

### 1. `num_iters` - 迭代次数（完整评估次数）
- **含义**: 在**完整数据集**上评估新程序的次数
- **与实际迭代数的关系**: 实际运行的迭代数通常 >> `num_iters`，因为大部分迭代只在 subsample 上运行
- **配置示例**:
  ```python
  GEPA-5:  num_iters=5   # 最多 5 次完整评估
  GEPA-10: num_iters=10  # 最多 10 次完整评估
  GEPA-50: num_iters=50  # 最多 50 次完整评估
  ```
- **实际效果**:
  - `num_iters=5` 可能实际运行 50-100+ 次迭代
  - 只有当 subsample 上有改进时，才会触发完整评估

### 2. `max_evals_per_trainval_instance` - 每样本评估上限
- **含义**: 平均每个训练/验证样本被评估的次数
- **计算公式**: `total_evals / train_val_size`
- **用途**: 控制总体计算成本
- **示例**:
  ```python
  train_val_size = 300
  max_evals_per_trainval_instance = 10
  → 最多进行 3000 次样本评估
  ```

### 3. `max_metric_calls` - 总评估调用上限
- **含义**: metric 函数的总调用次数
- **用途**: 直接限制计算成本
- **优先级**: 三个参数中只能设置一个
- **示例**:
  ```python
  max_metric_calls = 5000  # 最多评估 5000 个样本
  ```

### 4. `num_dspy_examples_per_gepa_step` - 每步采样数（Mini-batch 大小）
- **含义**: 每次迭代从训练集采样的样本数
- **默认值**: `3`
- **影响**:
  - 越小：迭代更快，但可能不够代表性
  - 越大：评估更准确，但计算成本更高
- **推荐值**: 3-25

## 优化参数

### 5. `use_merge` - 是否使用程序合并
- **含义**: 是否尝试合并优秀程序的 instructions
- **默认值**: `False`
- **开启条件**:
  ```python
  use_merge=True
  max_merge_invocations=5  # 最多尝试 5 次合并
  ```

### 6. `track_scores_on` - 评分跟踪策略
- **选项**:
  - `'val'`: 只在验证集上跟踪分数（推荐，避免过拟合）
  - `'train_val'`: 同时跟踪训练集和验证集
- **默认**: `'train_val'`

### 7. `set_for_merge_minibatch` - 合并时的采样集
- **选项**: `'train'`, `'val'`, `'both'`
- **含义**: 合并程序时从哪个集合采样来评估

## 实际配置示例

### 示例 1: 快速原型（5次完整评估）
```python
gepa = GEPA(
    num_iters=5,                           # 最多 5 次完整评估
    num_dspy_examples_per_gepa_step=3,     # 每步采样 3 个样本
    use_merge=False,                       # 不使用合并
    track_scores_on='val',                 # 只在验证集跟踪
)
```
**预期**:
- 实际迭代数: ~50-100 次
- 完整评估: 最多 5 次
- 计算成本: 低

### 示例 2: 标准训练（10次完整评估）
```python
gepa = GEPA(
    num_iters=10,
    num_dspy_examples_per_gepa_step=5,
    use_merge=True,
    max_merge_invocations=5,
    track_scores_on='val',
)
```
**预期**:
- 实际迭代数: ~100-200 次
- 完整评估: 最多 10 次
- 合并尝试: 最多 5 次
- 计算成本: 中等

### 示例 3: 充分优化（50次完整评估）
```python
gepa = GEPA(
    num_iters=50,
    num_dspy_examples_per_gepa_step=10,
    use_merge=True,
    max_merge_invocations=10,
    track_scores_on='val',
)
```
**预期**:
- 实际迭代数: ~500+ 次
- 完整评估: 最多 50 次
- 计算成本: 高

## 迭代流程详解

```
GEPA 执行流程:

初始化: 评估 base_program
↓
┌─────────────────────────────────────────┐
│ 迭代 N (轻量级)                          │
│ ├─ 采样 3 个样本                         │
│ ├─ 收集反馈                              │
│ ├─ 生成新 instruction                   │
│ ├─ 在 3 个样本上评估                     │
│ └─ 如果没改进 → 跳过，进入下一迭代        │
└─────────────────────────────────────────┘
       ↓ (如果改进了)
┌─────────────────────────────────────────┐
│ 完整评估 (计入 num_iters)                │
│ ├─ 在全部训练集上评估                    │
│ ├─ 在全部验证集上评估                    │
│ ├─ 添加到候选程序池                      │
│ └─ 更新 Pareto Front                    │
└─────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────┐
│ 程序合并 (可选)                          │
│ ├─ 选择 Pareto Front 上的优秀程序        │
│ ├─ 尝试合并它们的 instructions           │
│ └─ 评估合并后的程序                      │
└─────────────────────────────────────────┘
       ↓
继续下一次迭代，直到达到 num_iters 上限
```

## 计算成本估算

### 示例计算（GEPA-10）
假设:
- 训练集: 200 个样本
- 验证集: 100 个样本
- `num_iters=10`
- `num_dspy_examples_per_gepa_step=3`
- 实际找到 10 次改进（触发完整评估）
- 每次改进前平均尝试 10 次

**计算**:
```
初始评估:
  - 训练集: 200 次
  - 验证集: 100 次
  = 300 次

迭代采样评估:
  - 10 次成功 × 10 次尝试 × 3 个样本
  = 300 次

完整评估（10次改进）:
  - 10 × (200 + 100)
  = 3000 次

总计: 300 + 300 + 3000 = 3600 次样本评估
```

### 与直接遍历对比
如果不用 GEPA，直接评估所有可能的 instructions:
```
假设每个 predictor 尝试 20 个不同的 instructions
3 个 predictors × 20 个 instructions × 300 个样本
= 18,000 次样本评估
```

**GEPA 节省了约 80% 的计算成本！**

## 如何选择参数

### 快速实验（预算有限）
```python
num_iters=5
num_dspy_examples_per_gepa_step=3
```

### 正常训练（平衡性能和成本）
```python
num_iters=10-15
num_dspy_examples_per_gepa_step=5-10
use_merge=True
```

### 充分优化（追求最佳性能）
```python
num_iters=20-50
num_dspy_examples_per_gepa_step=10-25
use_merge=True
max_merge_invocations=10
```

### 大规模数据集（样本数 > 1000）
```python
# 使用 max_evals_per_trainval_instance 控制
max_evals_per_trainval_instance=10-20
num_dspy_examples_per_gepa_step=10-50
```

## 监控迭代进度

### 在日志中查看
```
Iteration 1: Selected program candidate 0 with base score: 0.70
Iteration 1: Updating predictor hop1
Iteration 1: New subsample score: 0.85
Iteration 1: New subsample score is better, updating program candidate!
→ 触发完整评估

Iteration 2: Selected program candidate 1 with base score: 0.75
Iteration 2: Updating predictor hop2
Iteration 2: New subsample score: 0.72
Iteration 2: New subsample score is not better, skipping
→ 不触发完整评估，继续下一迭代
```

### 在 WandB 中查看
- `iteration`: 当前迭代编号
- `num_full_ds_evals`: 已进行的完整评估次数
- `subsample_score`: 当前 subsample 分数
- `new_subsample_score`: 新程序的 subsample 分数
- `best_val_score`: 最佳验证集分数

## 总结

| 参数 | 含义 | 影响 |
|------|------|------|
| `num_iters` | 完整评估次数 | 直接控制优化深度 |
| `num_dspy_examples_per_gepa_step` | Mini-batch 大小 | 控制每次迭代的计算成本 |
| `use_merge` | 是否合并程序 | 可能找到更好的组合 |
| `track_scores_on` | 评分跟踪策略 | 影响过拟合风险 |

**关键理解**: `num_iters` 不是实际的迭代次数，而是**成功改进的次数上限**。大部分迭代都是轻量级的 subsample 评估，只有改进时才会触发昂贵的完整评估。
