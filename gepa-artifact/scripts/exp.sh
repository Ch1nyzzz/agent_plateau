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
export CUDA_VISIBLE_DEVICES=0,1

# Ray 配置：避免超时和警告
export RAY_DEDUP_LOGS=0
export RAY_BACKEND_LOG_LEVEL=warning
export PYTHONWARNINGS="ignore:os.fork:RuntimeWarning"
# 关闭 JAX（如果不需要）
export JAX_PLATFORMS=""

# ========================================
# 可配置参数 - 在这里修改实验配置
# ========================================

# GPU 配置
TENSOR_PARALLEL_SIZE=2      # 每个模型实例使用的 GPU 数量(张量并行)
NUM_MODEL_INSTANCES=1       # 模型实例数量(数据并行)

# vLLM 推理配置
GPU_MEMORY_UTIL=0.85        # GPU 显存利用率 (0.0-1.0) - 稍微增加
MAX_MODEL_LEN=8192          # 最大模型序列长度 - 降低以节省KV cache显存
MAX_NUM_SEQS=1            # 最大并行处理序列数 - 降低以节省显存
TEMPERATURE=0.6             # 采样温度
MAX_TOKENS=8192             # 最大生成 token 数
TOP_P=0.95                  # Top-p 采样参数

# 实验配置
BENCHMARK="aime"            # 基准测试: aime, lb_math, hover, papillon, hotpotqa, ifbench
NUM_EVAL_THREADS=1         # 评估线程数
NUM_OPTIMIZE_THREADS=1     # 优化线程数

# ========================================
# 运行实验
# ========================================

python scripts/exp.py \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --num-model-instances $NUM_MODEL_INSTANCES \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS \
    --top-p $TOP_P \
    --benchmark $BENCHMARK \
    --num-eval-threads $NUM_EVAL_THREADS \
    --num-optimize-threads $NUM_OPTIMIZE_THREADS

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
