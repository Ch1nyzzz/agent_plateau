# ACE-open 仓库结构与复现指南

本文档详细介绍了 ACE-open 仓库的结构和完整的复现步骤。

## 📋 项目概述

这是 **Agentic Context Engineering (ACE)** 方法的复现实现，基于论文 *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models* (arXiv:2510.04618)。

**重要说明**：这是一个**非官方复现版本**，因为论文官方代码尚未发布。

### 核心概念

**三角色系统**：
1. **Generator（生成器）** - 使用playbook生成答案，标记使用的bullet IDs
2. **Reflector（反思器）** - 分析错误和反馈，最多5轮迭代精化
3. **Curator（策展器）** - 基于反思产生playbook增量更新

**核心数据结构**：
- **Playbook** - 结构化上下文存储，由多个bullets组成
- **Bullet** - 单个策略条目，包含ID、section、content和helpful/harmful/neutral计数器
- **Delta** - 增量更新操作，支持ADD/UPDATE/TAG/REMOVE四种类型

### 仓库统计

- **仓库大小**: 480KB
- **核心代码**: 1046行 Python 代码
- **主要文件数**: 15个（不含git和缓存）
- **支持Python版本**: 3.9+（开发环境使用3.12）

---

## 🏗️ 完整目录结构

```
/home/yuhan/ReAct_learning/ACE-open/
├── README.md                          # 项目主文档与快速开始指南
├── questions.json                     # 10个火灾调查问题的样本数据集
│
├── ace/                               # 核心库模块 (1046行代码)
│   ├── __init__.py                    # 模块导出接口（暴露所有公共API）
│   ├── playbook.py                    # Playbook存储与结构化上下文管理 (216行)
│   ├── delta.py                       # 增量更新操作定义 (68行)
│   ├── llm.py                         # LLM客户端抽象与实现 (170行)
│   ├── roles.py                       # 三个角色实现 (271行)
│   ├── prompts.py                     # 角色提示词模板 (90行)
│   ├── adaptation.py                  # 离线/在线适应循环驱动 (194行)
│   └── __pycache__/                   # Python字节码缓存
│
├── docs/                              # 文档目录
│   └── method_outline.md              # 论文方法总结与实现细节
│
├── tests/                             # 单元测试
│   ├── test_adaptation.py             # 完整的适应循环回归测试 (100行)
│   └── __pycache__/                   # 缓存
│
├── scripts/                           # 可执行脚本
│   ├── run_local_adapter.py           # 最小化的本地模型ACE循环演示 (137行)
│   ├── run_questions.py               # 完整的问题测试与报告生成 (267行)
│   └── run_questions_direct.py        # 无ACE的直接LLM基线对照 (180行)
│
├── reports/                           # 输出报告
│   └── questions_report.md            # 生成的测试结果报告
│
└── .git/                              # Git版本控制
```

---

## 📚 核心模块详解

### 1. ace/playbook.py (216行)

**作用**: Playbook结构化存储管理

**关键类**:

#### Bullet
单个playbook条目，包含:
- `id`: 唯一标识符
- `section`: 所属分类（如"defaults", "error_handling"等）
- `content`: 具体策略文本
- `helpful/harmful/neutral`: 计数器（追踪反馈信号）
- `created_at/updated_at`: 时间戳

#### Playbook
Playbook数据结构，核心方法：
- `add_bullet()`: 添加新bullet
- `update_bullet()`: 更新现有bullet
- `tag_bullet()`: 调整反馈计数器
- `remove_bullet()`: 删除bullet
- `apply_delta()`: 应用策展器产生的增量更新
- `as_prompt()`: 生成可用于LLM的文本格式
- JSON序列化/反序列化支持

### 2. ace/delta.py (68行)

**作用**: 增量操作的定义和序列化

**关键类**:

#### DeltaOperation
单个变更操作，支持4种类型:
- `ADD`: 添加新bullet
- `UPDATE`: 更新现有bullet
- `TAG`: 调整helpful/harmful/neutral计数
- `REMOVE`: 删除bullet

#### DeltaBatch
一批curator操作的集合:
- `reasoning`: 策展器的推理说明
- `operations`: 具体操作列表
- JSON序列化支持

### 3. ace/llm.py (170行)

**作用**: LLM客户端抽象层与实现

**关键类**:

#### LLMClient (抽象基类)
定义通用接口:
- `complete(prompt, **kwargs)`: 返回`LLMResponse`

#### DummyLLMClient
测试用的确定性LLM:
- 使用队列预注入响应
- 用于单元测试（无需实际模型）

#### TransformersLLMClient
本地transformers模型客户端:
- 支持chat格式提示
- 配置参数:
  - `max_new_tokens`: 最大生成令牌数
  - `temperature`: 采样温度（0表示贪心）
  - `torch_dtype`: 精度设置（bfloat16推荐）
  - `device_map`: GPU分配策略
- JSON后处理: 自动提取和清理JSON格式响应

### 4. ace/roles.py (271行)

**作用**: 实现三个主要角色

**关键类**:

#### Generator - 生成答案
- **输入**: 当前playbook + 问题 + 反射反馈
- **输出**: `GeneratorOutput`
  - `reasoning`: 推理过程
  - `final_answer`: 最终答案
  - `bullet_ids`: 使用的bullet ID列表
- **特性**: 提示重试机制（JSON解析失败时自动重试，最多3次）

#### Reflector - 分析错误与反馈
- **输入**: 生成输出 + 环境反馈 + 真实标签
- **输出**: `ReflectorOutput`
  - `error_identification`: 错误识别
  - `root_cause_analysis`: 根本原因
  - `correct_approach`: 纠正方案
  - `key_insight`: 可重用的洞察
  - `bullet_tags`: [(bullet_id, tag)对] - 标记哪些bullet有帮助/有害
- **特性**: 支持迭代精化（最多5轮）

#### Curator - 产生playbook更新
- **输入**: 反射输出 + 当前playbook
- **输出**: `CuratorOutput`
  - `delta`: DeltaBatch 增量操作
- **特性**: 在令牌预算意识下工作，避免playbook膨胀

### 5. ace/prompts.py (90行)

**作用**: 三角色的提示词模板

**包含**:
- `GENERATOR_PROMPT`: 指导Generator生成答案
- `REFLECTOR_PROMPT`: 指导Reflector分析
- `CURATOR_PROMPT`: 指导Curator编辑playbook
- **所有提示词都要求JSON输出格式**

### 6. ace/adaptation.py (194行)

**作用**: 离线和在线适应循环的驱动

**关键类**:

#### Sample
单个任务实例:
- `question`: 输入问题
- `context`: 可选上下文
- `ground_truth`: 可选真实标签
- `metadata`: 自定义元数据

#### EnvironmentResult
环境反馈:
- `feedback`: 文本反馈
- `ground_truth`: 真实答案
- `metrics`: 量化指标（如准确率）

#### TaskEnvironment (抽象基类)
任务评估接口:
- `evaluate(sample, generator_output)`: 返回环境反馈

#### AdapterBase
共享适应逻辑:
- 管理playbook、三个角色、最近反射缓存
- `_process_sample()`: 处理单个样本的完整流程
  1. Generator生成答案
  2. Environment评估
  3. Reflector分析
  4. Curator产生delta
  5. 应用delta到playbook

#### OfflineAdapter
多轮离线训练:
- `run(samples, environment, epochs)`: 在多个epoch上迭代训练
- 论文中使用最多5个epoch

#### OnlineAdapter
流式在线学习:
- `run(samples, environment)`: 逐样本处理，动态更新playbook

---

## 📜 可执行脚本详解

### 1. scripts/run_local_adapter.py (137行)

**目标**: 展示如何使用本地transformers模型运行ACE

**功能**:
- 使用TransformersLLMClient加载本地模型
- 创建单个样本进行演示
- 执行一个epoch的OfflineAdapter循环
- 打印所有中间步骤的JSON输出（Generator、Reflector、Curator）
- 打印最终playbook状态

**命令行参数**:
- `--model-path`: 模型权重目录（默认/data/models/openai/gpt-oss-20b）
- `--cuda-visible-devices`: GPU设备（默认2,3）
- `--max-new-tokens`: 最大生成令牌数（默认512）
- `--temperature`: 采样温度（默认0.0）

**用途**: 快速验证ACE框架是否正确连接

### 2. scripts/run_questions.py (267行)

**目标**: 在问题数据集上执行完整的ACE适应，生成详细报告

**功能**:
- 加载questions.json中的10个火灾调查问题
- 创建FireInvestigationEnvironment（使用字符串相似度评估）
- 执行多epoch OfflineAdapter循环
- 收集相似度指标（平均/最小/最大）
- 生成Markdown报告，包含:
  - 整体统计（样本数、相似度分布）
  - 每个问题的摘要表格
  - 详细的Q&A对比
  - 每一步的Reflector反射和Curator操作
  - 最终playbook状态

**命令行参数**:
- `--model-path`: 模型路径（默认/data/models/openai/gpt-oss-20b）
- `--questions`: 问题文件（默认questions.json）
- `--output`: 报告输出路径（默认reports/questions_report.md）
- `--cuda-visible-devices`: GPU设备（默认2,3）
- `--epochs`: 训练epoch数（默认1）
- `--max-new-tokens`: 最大令牌数（默认512）
- `--temperature`: 采样温度（默认0.0）
- `--similarity-threshold`: 相似度阈值（默认0.7）

**输出**: 详细的Markdown报告

### 3. scripts/run_questions_direct.py (180行)

**目标**: 无ACE的直接LLM调用作为基线对比

**功能**:
- 加载同一问题数据集
- 对每个问题直接调用LLM（不使用ACE）
- 计算与参考答案的字符串相似度
- 生成类似格式的Markdown报告

**用途**: 衡量ACE带来的性能改进

---

## 🚀 复现步骤

### 步骤 1️⃣：环境准备

#### Python版本要求

```bash
# 需要 Python 3.9+（开发使用3.12）
python --version
```

#### 创建虚拟环境（可选但推荐）

```bash
cd /home/yuhan/ReAct_learning/ACE-open
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

#### 依赖说明

- ✅ **核心库无第三方依赖** - ace模块可独立运行
- ⚠️ **本地模型需要** `transformers`, `torch`（使用TransformersLLMClient时）

如果需要使用本地模型，安装依赖：

```bash
pip install torch transformers accelerate
```

### 步骤 2️⃣：运行单元测试（验证安装）

```bash
# 在仓库根目录运行
python -m unittest discover -s tests
```

**预期输出**:

```
test_single_step_updates_playbook (test_adaptation.OfflineAdapterTest) ... ok
----------------------------------------------------------------------
Ran 1 test in 0.XXXs
OK
```

### 步骤 3️⃣：快速演示（使用DummyLLMClient）

创建测试脚本 `quick_demo.py`：

```python
import json
from ace import (
    Playbook, DummyLLMClient, Generator, Reflector, Curator,
    OfflineAdapter, Sample, TaskEnvironment, EnvironmentResult
)

class ToyEnv(TaskEnvironment):
    def evaluate(self, sample, generator_output):
        gt = sample.ground_truth or ""
        pred = generator_output.final_answer
        feedback = "correct" if pred == gt else f"expected {gt} but got {pred}"
        return EnvironmentResult(feedback=feedback, ground_truth=gt)

# 创建Dummy LLM客户端（预注入响应）
client = DummyLLMClient()
client.queue(json.dumps({
    "reasoning": "The answer is 42",
    "bullet_ids": [],
    "final_answer": "42"
}))
client.queue(json.dumps({
    "reasoning": "Analysis complete",
    "error_identification": "",
    "root_cause_analysis": "",
    "correct_approach": "",
    "key_insight": "Always remember 42.",
    "bullet_tags": []
}))
client.queue(json.dumps({
    "reasoning": "Adding default rule",
    "operations": [{
        "type": "ADD",
        "section": "defaults",
        "content": "Answer 42 when uncertain.",
        "metadata": {"helpful": 1}
    }]
}))

# 创建适配器
adapter = OfflineAdapter(
    playbook=Playbook(),
    generator=Generator(client),
    reflector=Reflector(client),
    curator=Curator(client),
)

# 运行适应循环
samples = [Sample(question="What is the meaning of life?", ground_truth="42")]
results = adapter.run(samples, ToyEnv(), epochs=1)

print("Playbook after adaptation:")
print(adapter.playbook.as_prompt())
```

运行：

```bash
python quick_demo.py
```

### 步骤 4️⃣：使用本地模型运行（最小演示）

**配置模型路径**：
编辑 `scripts/run_local_adapter.py` 或使用命令行参数

**运行示例**：

```bash
# 设置GPU设备
CUDA_VISIBLE_DEVICES=2,3 python scripts/run_local_adapter.py \
    --model-path /data/models/openai/gpt-oss-20b \
    --max-new-tokens 512 \
    --temperature 0.0
```

**参数说明**：
- `--model-path`: 本地模型权重目录
- `--cuda-visible-devices`: GPU设备号（默认2,3）
- `--max-new-tokens`: 最大生成令牌数
- `--temperature`: 采样温度（0=确定性）

### 步骤 5️⃣：完整实验（火灾调查问题集）

#### 运行ACE适应循环

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/run_questions.py \
    --model-path /data/models/openai/gpt-oss-20b \
    --questions questions.json \
    --output reports/questions_report.md \
    --epochs 5 \
    --max-new-tokens 512 \
    --temperature 0.0 \
    --similarity-threshold 0.7
```

**参数详解**：
- `--epochs`: 离线训练轮数（论文中最多5轮）
- `--similarity-threshold`: 环境反馈阈值（默认0.7）
- `--output`: Markdown报告输出路径

#### 运行基线对照（无ACE）

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/run_questions_direct.py \
    --model-path /data/models/openai/gpt-oss-20b \
    --questions questions.json \
    --output reports/baseline_report.md
```

#### 对比报告

```bash
# 查看ACE报告
cat reports/questions_report.md

# 查看基线报告
cat reports/baseline_report.md
```

### 步骤 6️⃣：扩展到自定义任务

#### 实现自定义LLM客户端

```python
from ace import LLMClient, LLMResponse

class MyLLMClient(LLMClient):
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        # 调用你的LLM API
        response = my_llm_api.generate(prompt, **kwargs)
        return LLMResponse(
            content=response.text,
            metadata={"tokens": response.tokens}
        )
```

#### 实现任务特定环境

```python
from ace import TaskEnvironment, EnvironmentResult

class MyTaskEnvironment(TaskEnvironment):
    def evaluate(self, sample, generator_output):
        # 执行任务并获取反馈
        result = execute_task(generator_output.final_answer)
        feedback = analyze_result(result)
        metrics = {"accuracy": result.score}

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=sample.ground_truth,
            metrics=metrics
        )
```

#### 配置适应器

```python
from ace import OfflineAdapter, OnlineAdapter

# 离线适应（多epoch训练）
offline_adapter = OfflineAdapter(
    playbook=Playbook(),
    generator=Generator(my_client),
    reflector=Reflector(my_client),
    curator=Curator(my_client),
    max_refinement_rounds=5  # 论文设置
)

results = offline_adapter.run(
    samples=training_samples,
    environment=my_env,
    epochs=5  # 论文设置
)

# 在线适应（测试时学习）
online_adapter = OnlineAdapter(
    playbook=results_playbook,  # 使用离线训练的playbook
    generator=Generator(my_client),
    reflector=Reflector(my_client),
    curator=Curator(my_client)
)

online_results = online_adapter.run(
    samples=test_samples,
    environment=my_env
)
```

---

## 📊 论文中的关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 基础模型 | DeepSeek-V3.1 | 非thinking模式 |
| 批大小 | 1 | 每个样本一个delta |
| Reflector精化轮数 | 最多5轮 | 迭代改进反思 |
| 离线epoch数 | 最多5轮 | 多轮训练 |
| 在线模式 | 顺序处理 | 逐样本更新 |

---

## 🔄 工作流程详解

### 离线适应循环（Offline Adaptation）

```
Input: samples[], environment, epochs
  └─ For each epoch:
      └─ For each sample in samples:
          1. Generator.generate()
             Input: question + playbook + reflection_context
             Output: GeneratorOutput (reasoning, final_answer, bullet_ids)

          2. Environment.evaluate()
             Input: sample + generator_output
             Output: EnvironmentResult (feedback, metrics)

          3. Reflector.reflect()
             Input: question + generator_output + playbook + feedback
             Output: ReflectorOutput (insights, bullet_tags)

          4. Curator.curate()
             Input: reflection + playbook + progress
             Output: CuratorOutput (DeltaBatch with operations)

          5. Playbook.apply_delta()
             Input: DeltaBatch
             Updates playbook in-place (ADD/UPDATE/TAG/REMOVE)

Output: [AdapterStepResult, ...] + final playbook
```

### 在线适应循环（Online Adaptation）

```
Input: samples[], environment
  └─ For each sample in samples:
      1. Generator生成答案（使用当前playbook）
      2. Environment评估并返回反馈
      3. Reflector分析错误（最多5轮精化）
      4. Curator产生Delta更新
      5. 应用Delta到Playbook
      6. 使用更新后的playbook处理下一个样本

Output: [AdapterStepResult, ...] + evolved playbook
```

---

## 🗂️ 关键文件索引

| 文件/目录 | 行数 | 主要作用 | 关键输入/输出 |
|---------|------|--------|-----------|
| `ace/playbook.py` | 216 | Playbook结构化存储 | Bullet管理、JSON序列化、Delta应用 |
| `ace/delta.py` | 68 | 增量操作定义 | DeltaOperation、DeltaBatch |
| `ace/llm.py` | 170 | LLM客户端抽象 | LLMClient接口、DummyLLMClient、TransformersLLMClient |
| `ace/roles.py` | 271 | 三角色实现 | Generator、Reflector、Curator |
| `ace/prompts.py` | 90 | 提示词模板 | 3个角色的系统提示词 |
| `ace/adaptation.py` | 194 | 适应循环驱动 | OfflineAdapter、OnlineAdapter、环境接口 |
| `tests/test_adaptation.py` | 100 | 集成测试 | 完整适应循环验证 |
| `scripts/run_questions.py` | 267 | 主测试脚本 | 完整实验执行与报告生成 |
| `scripts/run_local_adapter.py` | 137 | 最小演示 | 本地模型集成示例 |
| `scripts/run_questions_direct.py` | 180 | 基线对比 | 无ACE的直接LLM调用 |
| `docs/method_outline.md` | 64 | 方法文档 | 论文方法总结 |
| `questions.json` | - | 样本数据集 | 10个专业火灾调查问题 |

### 代码路径参考

- **Playbook核心逻辑**: `ace/playbook.py:1-216`
- **Delta操作定义**: `ace/delta.py:1-68`
- **三角色实现**: `ace/roles.py:1-271`
- **适应循环驱动**: `ace/adaptation.py:1-194`
- **主测试脚本**: `scripts/run_questions.py:1-267`

---

## 🎨 架构特点

### 模块化设计
- 清晰的抽象接口（LLMClient、TaskEnvironment）
- 三角色分离（职责单一）
- Playbook是数据中心，所有操作围绕它

### 扩展性
- 易于实现新的LLMClient（如OpenAI、Claude等）
- 易于实现特定领域的TaskEnvironment
- 支持自定义提示词

### 实验友好
- 包含DummyLLMClient用于测试
- 完整的单元测试
- 脚本示例展示多种使用方式

### 论文实现忠实度
- 遵循论文中的三个角色定义
- 支持多epoch离线训练
- 支持在线流式学习
- Playbook计数器追踪helpful/harmful/neutral反馈

---

## ⚠️ 注意事项

### 关于实现

1. **非官方实现** - 这是根据论文复现的版本，官方代码发布后可能有差异
2. **无硬性依赖** - 核心ace库不依赖第三方包，可用DummyLLMClient测试
3. **Token预算** - Curator在token预算意识下工作，避免playbook膨胀

### 运行要求

1. **模型要求** - 如使用本地模型，需要足够的GPU内存（建议>=16GB VRAM）
2. **反馈质量** - ACE性能依赖可靠的环境反馈信号
3. **计算资源** - 多epoch训练和反思精化需要较多计算时间

### 数据集

1. **样本数据** - questions.json包含10个火灾调查专业问题
2. **评估指标** - 当前使用字符串相似度（SequenceMatcher）
3. **自定义数据** - 可以替换为任何JSON格式的问题集

---

## ✅ 复现检查清单

环境配置：
- [ ] Python 3.9+ 已安装
- [ ] 单元测试通过 (`python -m unittest discover -s tests`)
- [ ] DummyLLMClient演示成功运行

本地模型（可选）：
- [ ] transformers和torch已安装
- [ ] 本地模型路径已配置
- [ ] GPU设备可用并正确设置

理解与扩展：
- [ ] 理解三角色工作流程
- [ ] 理解Playbook和Delta机制
- [ ] 可以实现自定义LLMClient
- [ ] 可以实现自定义TaskEnvironment

完整实验：
- [ ] 成功运行run_local_adapter.py
- [ ] 成功运行run_questions.py
- [ ] 成功运行基线对照run_questions_direct.py
- [ ] 能够对比ACE和基线性能

---

## 📖 额外资源

### 相关文档
- **论文**: *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models* (arXiv:2510.04618)
- **方法总结**: `docs/method_outline.md`
- **主README**: `README.md`

### 学习路径建议

1. **理解核心概念** - 阅读 `docs/method_outline.md`
2. **运行单元测试** - 理解基本工作流程
3. **查看DummyLLM示例** - 学习如何使用API
4. **运行本地模型演示** - 体验完整ACE循环
5. **分析问题集结果** - 对比ACE和基线性能
6. **实现自定义任务** - 扩展到你的领域

### 故障排除

**问题：单元测试失败**
- 检查Python版本是否>=3.9
- 确认在仓库根目录运行测试

**问题：模型加载失败**
- 检查模型路径是否正确
- 确认GPU设备是否可用
- 检查transformers版本是否兼容

**问题：JSON解析错误**
- 检查LLM输出格式是否符合预期
- 查看提示词是否正确指定JSON格式
- 调整max_new_tokens参数

**问题：性能不佳**
- 增加epochs数量（最多5）
- 调整reflector的max_refinement_rounds
- 检查环境反馈信号是否有效
- 确认使用的模型能力是否足够

---

## 🤝 贡献与反馈

这是一个社区复现项目，如有问题或改进建议：
1. 检查论文原文以确认实现细节
2. 查看issues和讨论
3. 提交bug报告或功能请求
4. 参与代码审查和优化

---

**最后更新**: 2025-10-17
**文档版本**: 1.0