#!/bin/bash
# ========================================
# GEPA Experiment - Ray + vLLM Optimized
# 针对 4 卡 GPU 的最优配置
# ========================================

cd /home/yuhan/ReAct_learning/gepa-artifact
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
# 最优配置 - 4 卡 GPU + 8B 模型
# ========================================
# Tensor Parallel: 1 (每个模型实例用 1 张 GPU，8B 模型单卡足够)
# Data Parallel: 4 (运行 4 个模型实例，最大化吞吐量)
# 预期吞吐量提升: ~4x

python scripts/exp.py \
    --tensor-parallel-size 1 \
    --num-model-instances 2 \
    --benchmark aime \
    --num-eval-threads 64 \
    --num-optimize-threads 80

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
