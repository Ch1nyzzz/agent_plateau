# 批量实验运行指南

已为您创建了两个批量运行脚本：Bash 版本和 Python 版本。

## 📁 文件说明

- **`run_batch_experiments.sh`** - Bash 脚本版本，简单易用
- **`run_batch_experiments.py`** - Python 脚本版本，功能更丰富
- **`batch_experiment_logs/`** - 日志目录（自动创建）

## 🚀 快速开始

### 方法 1: 使用 Bash 脚本（推荐新手）

```bash
# 顺序运行所有实验
bash run_batch_experiments.sh

# 从第 6 个实验开始
bash run_batch_experiments.sh --start-from 6

# 并行运行（谨慎！会同时运行所有实验）
bash run_batch_experiments.sh --parallel
```

### 方法 2: 使用 Python 脚本（推荐进阶用户）

```bash
# 顺序运行所有实验
python run_batch_experiments.py

# 从第 6 个实验开始
python run_batch_experiments.py --start-from 6

# 只运行特定实验（例如：1, 3, 5）
python run_batch_experiments.py --only 1,3,5

# 跳过某些实验（例如：2, 4）
python run_batch_experiments.py --skip 2,4

# 预览要运行的实验（不实际执行）
python run_batch_experiments.py --dry-run

# 并行运行（谨慎！）
python run_batch_experiments.py --parallel
```

## 📋 实验列表

脚本会按顺序运行以下 12 个实验：

### HoverBench (5 个实验)
1. GEPA-10
2. GEPA-15
3. GEPA-20
4. GEPA-25
5. GEPA-50

### HotpotQABench (7 个实验)
6. Baseline
7. GEPA-5
8. GEPA-10
9. GEPA-15
10. GEPA-20
11. GEPA-25
12. GEPA-50

## 📊 日志和结果

### 日志文件位置
```
batch_experiment_logs/
├── batch_run_20250102_143000.log          # 主日志
├── exp_1_hoverBench_..._20250102_143000.log  # 实验1日志
├── exp_2_hoverBench_..._20250102_143000.log  # 实验2日志
├── ...
└── experiment_summary_20250102_143000.txt     # 结果摘要
```

### 查看进度
```bash
# 查看主日志（实时）
tail -f batch_experiment_logs/batch_run_*.log

# 查看特定实验的日志
tail -f batch_experiment_logs/exp_1_*.log

# 查看摘要
cat batch_experiment_logs/experiment_summary_*.txt
```

## 💡 使用建议

### 顺序运行（推荐）
```bash
# 稳妥方案：按顺序运行，出错时可以选择继续或停止
python run_batch_experiments.py
```

**优点**：
- ✅ 资源占用可控
- ✅ 出错时可以手动干预
- ✅ 日志清晰，易于调试

**缺点**：
- ❌ 总耗时长（需要等待每个实验完成）

### 并行运行（仅限资源充足时）
```bash
# 激进方案：同时运行多个实验
python run_batch_experiments.py --parallel
```

**优点**：
- ✅ 总耗时短（实验并行执行）

**缺点**：
- ❌ 资源占用高（CPU、内存、API 配额）
- ❌ 可能触发 API 速率限制
- ❌ 出错后难以恢复

**建议仅在以下情况使用并行模式**：
- 有充足的 GPU/CPU 资源
- API 配额足够大
- 了解如何处理并发错误

### 分阶段运行
```bash
# 第一阶段：运行 HoverBench 实验 (1-5)
python run_batch_experiments.py --only 1,2,3,4,5

# 第二阶段：运行 HotpotQA Baseline
python run_batch_experiments.py --only 6

# 第三阶段：运行 HotpotQA GEPA 实验 (7-12)
python run_batch_experiments.py --only 7,8,9,10,11,12
```

## ⚠️ 注意事项

### 1. 检查环境变量
确保设置了 `OPENAI_API_KEY`：
```bash
echo $OPENAI_API_KEY
```

如果没有设置：
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. 预估时间和成本

单个实验耗时（粗略估计）：
- Baseline: 10-30 分钟
- GEPA-5: 30-60 分钟
- GEPA-10: 1-2 小时
- GEPA-15: 2-3 小时
- GEPA-20: 3-4 小时
- GEPA-25: 4-5 小时
- GEPA-50: 8-12 小时

**总耗时估计（顺序运行）**：
- 所有 12 个实验：**约 40-80 小时**

**建议**：
- 使用 tmux 或 screen 在后台运行
- 分批运行，不要一次运行所有实验
- 先运行较小的实验（GEPA-5, GEPA-10）测试配置

### 3. 使用 tmux 持久化运行
```bash
# 创建新的 tmux 会话
tmux new -s gepa_experiments

# 在 tmux 中运行脚本
python run_batch_experiments.py

# 分离会话（Ctrl+B 然后按 D）
# 或直接关闭终端，脚本会继续运行

# 重新连接到会话
tmux attach -t gepa_experiments

# 查看所有会话
tmux ls
```

### 4. 中断恢复

如果脚本中断，可以从特定实验继续：
```bash
# 假设实验 1-5 已完成，从实验 6 开始
python run_batch_experiments.py --start-from 6
```

### 5. 监控资源使用
```bash
# 监控 GPU 使用（如果使用 GPU）
watch -n 1 nvidia-smi

# 监控 CPU 和内存
htop

# 监控磁盘空间
df -h

# 监控网络流量（可选）
iftop
```

## 🔍 故障排查

### 问题 1: 权限不足
```bash
# 添加执行权限
chmod +x run_batch_experiments.sh
chmod +x run_batch_experiments.py
```

### 问题 2: Python 脚本找不到模块
```bash
# 确保在项目根目录运行
cd /data/home/yuhan/ReAct_learning/agent_plateau/gepa-artifact
python run_batch_experiments.py
```

### 问题 3: API 速率限制
如果遇到 API 速率限制错误：
- 降低 `--num_threads` 参数
- 使用更高级别的 API key
- 在实验之间添加延迟

### 问题 4: 内存不足
如果遇到内存不足：
- 减少 `--num_threads` 参数
- 不要使用并行模式
- 关闭其他占用内存的程序

### 问题 5: 查看特定实验失败原因
```bash
# 查看最新的实验日志
ls -lt batch_experiment_logs/exp_*.log | head -1 | awk '{print $NF}'

# 查看特定实验的完整日志
cat batch_experiment_logs/exp_6_HotpotQABench_HotpotMultiHop_Baseline_*.log
```

## 📈 查看结果

### 自动生成的摘要
脚本运行结束后会自动生成摘要文件：
```bash
cat batch_experiment_logs/experiment_summary_*.txt
```

### 手动提取分数
```bash
# 提取所有实验的分数
grep -r "Average Metric:" batch_experiment_logs/exp_*.log
```

### 生成对比表格
可以使用以下 Python 脚本提取并对比结果：
```python
import re
from pathlib import Path

log_dir = Path("batch_experiment_logs")

results = {}
for log_file in log_dir.glob("exp_*.log"):
    with open(log_file) as f:
        content = f.read()
        match = re.search(r'Average Metric: (\d+) / (\d+) \(([\d.]+)%\)', content)
        if match:
            exp_name = log_file.stem.split('_', 2)[2]  # 提取实验名称
            score = float(match.group(3))
            results[exp_name] = score

# 打印结果
for exp, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{exp}: {score:.2f}%")
```

## 🎯 最佳实践

1. **分批运行**
   ```bash
   # 先运行快速实验测试配置
   python run_batch_experiments.py --only 1,6,7

   # 确认无误后运行剩余实验
   python run_batch_experiments.py --skip 1,6,7
   ```

2. **使用预览模式**
   ```bash
   # 先预览，确认实验列表正确
   python run_batch_experiments.py --dry-run
   ```

3. **定期检查**
   ```bash
   # 每小时检查一次进度
   watch -n 3600 "tail -20 batch_experiment_logs/batch_run_*.log"
   ```

4. **保存结果**
   ```bash
   # 实验完成后备份日志
   tar -czf experiment_results_$(date +%Y%m%d).tar.gz batch_experiment_logs/
   ```

## 📞 获取帮助

```bash
# Bash 脚本帮助（脚本内查看注释）
head -20 run_batch_experiments.sh

# Python 脚本帮助
python run_batch_experiments.py --help
```

---

祝实验顺利！🚀
