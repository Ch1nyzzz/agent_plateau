#!/bin/bash
# ========================================
# GEPA Experiment - Ray + vLLM Optimized
# ========================================
# 取消 RAY_ADDRESS 以确保启动新的集群而不是连接到现有集群
unset RAY_ADDRESS
export VLLM_TORCH_COMPILE=0
cd /home/yuhan/ReAct_learning/agent_plateau/gepa-artifact
source /home/yuhan/cyh_dev/bin/activate

# ========================================
# 环境变量配置
# ========================================
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V1=0  # 使用 vLLM V0 引擎（更稳定，显存管理更好）
export CUDA_VISIBLE_DEVICES=2,3

# Ray 配置：避免超时和警告
export RAY_DEDUP_LOGS=0
export RAY_BACKEND_LOG_LEVEL=warning
export PYTHONWARNINGS="ignore:os.fork:RuntimeWarning"
# 关闭 JAX（如果不需要）
export JAX_PLATFORMS=""
export JAX_PLATFORM_NAME=cpu
# ========================================
# 可配置参数 - 在这里修改实验配置
# ========================================

# GPU 配置
TENSOR_PARALLEL_SIZE=2      # 每个模型实例使用的 GPU 数量(张量并行)
NUM_MODEL_INSTANCES=1       # 模型实例数量(数据并行)
NUM_ROLLOUTS=3              # 每轮采样数量
# vLLM 推理配置
GPU_MEMORY_UTIL=0.9       # GPU 显存利用率 (0.0-1.0) - 降低为 KV cache 预留空间
MAX_NUM_SEQS=64            # 最大并行处理序列数 - 降低以节省显存
MAX_MODEL_LEN=16384        # 最大模型序列长度 - 降低以节省KV cache显存
TEMPERATURE=0.6             # 采样温度
MAX_TOKENS=8192             # 最大生成 token 数
TOP_P=0.95                  # Top-p 采样参数

# 实验配置
BENCHMARK="hotpotqa"            # 基准测试: aime, lb_math, hover, papillon, hotpotqa, ifbench
NUM_EVAL_THREADS=32         # 评估线程数
NUM_OPTIMIZE_THREADS=16     # 优化线程数
use_simple=0
# ========================================
# 运行实验
# ========================================

# 创建日志目录
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# 生成日志文件名（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/exp_${BENCHMARK}_${TIMESTAMP}.log"

echo "========================================"
echo "实验日志保存到: $LOG_FILE"
echo "优化配置:"
echo "  - MAX_NUM_SEQS: $MAX_NUM_SEQS (批量推理加速)"
echo "  - NUM_EVAL_THREADS: $NUM_EVAL_THREADS"
echo "  - NUM_OPTIMIZE_THREADS: $NUM_OPTIMIZE_THREADS"
echo "========================================"
echo ""

# 运行实验并保存日志（后台运行）
nohup python scripts/exp.py \
    --use-simple $use_simple \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --num-model-instances $NUM_MODEL_INSTANCES \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --num-rollouts $NUM_ROLLOUTS \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-model-len $MAX_MODEL_LEN \
    --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS \
    --top-p $TOP_P \
    --benchmark $BENCHMARK \
    --num-eval-threads $NUM_EVAL_THREADS \
    --num-optimize-threads $NUM_OPTIMIZE_THREADS \
    >> $LOG_FILE 2>&1 &

# 获取后台进程 PID
PID=$!

echo "✓ 实验已在后台启动！"
echo "  进程 PID: $PID"
echo "  日志文件: $LOG_FILE"
echo ""
echo "查看实时日志: tail -f $LOG_FILE"
echo "查看进程状态: ps -p $PID"
echo "停止实验: kill $PID"
echo ""

# 保存 PID 到文件方便后续管理
echo $PID > ./logs/exp_${BENCHMARK}.pid
echo "PID 已保存到: ./logs/exp_${BENCHMARK}.pid"

# ========================================
# 其他配置示例
# ========================================

# 【调试模式】快速测试（只用 2 个实例）
# python scripts/exp.py \
#     --tensor-parallel-size 1 \
#     --num-model-instances 2 \
#     --benchmark aime \
#     --debug \
#     --skip-optimize

# 【跳过基础评估】直接优化
# python scripts/exp.py \
#     --tensor-parallel-size 1 \
#     --num-model-instances 4 \
#     --benchmark aime \
#     --skip-base-eval

# 【多轮采样】提升性能
# python scripts/exp.py \
#     --tensor-parallel-size 1 \
#     --num-model-instances 4 \
#     --benchmark aime \
#     --num-rollouts 3

# 【大模型配置】如果使用 30B-70B 模型
# python scripts/exp.py \
#     --tensor-parallel-size 2 \
#     --num-model-instances 2 \
#     --benchmark aime
