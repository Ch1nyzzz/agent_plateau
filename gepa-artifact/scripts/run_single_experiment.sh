#!/bin/bash
# ============================================================================
# GEPA 单实验运行脚本 (基于 run_experiments.py)
# ============================================================================
# 用途: 运行单个实验配置，便于快速测试和调试
# 使用方法: bash scripts/run_single_experiment.sh
# ============================================================================

set -e  # 遇到错误立即退出

# ============================================================================
# 【必须配置】环境和路径设置
# ============================================================================
# 项目根目录（自动检测，或手动修改）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Python 环境（根据你的环境修改）
PYTHON_ENV="/home/yuhan/cyh_dev/bin/activate"
source "$PYTHON_ENV"

# ============================================================================
# 【核心配置】实验参数 - 在这里修改你的实验设置
# ============================================================================

# --- 基准测试选择 ---
# 可选值:
#   0=HoVer, 1=HotpotQA, 2=Papillon, 3=IFBench, 4=LiveBenchMath, 5=AIME
BENCHMARK_IDX=1                    # 基准测试索引
BENCHMARK_NAME="HotpotQABench"      # 基准测试名称（需与代码中一致）

# --- 程序选择 ---
PROGRAM_IDX=0                       # 程序索引（通常为0）
PROGRAM_NAME="HotpotMultiHop"       # 程序名称（需与代码中一致）

# --- 优化器选择 ---
# 可选值:
#   0=Baseline（无优化）
#   1=MIPROv2-Heavy（重度提示优化）
#   2=GEPA-MERGE（GEPA合并版本）
#   3=GEPA（标准GEPA）
#   4=Abl-SelectBestCandidate（消融研究）
#   5=GRPO（强化学习）
OPTIMIZER_IDX=0                     # 优化器索引
OPTIMIZER_NAME="Baseline"           # 优化器名称

# --- 语言模型配置 ---
# 在 experiment_configs.py 的 LM_CONFIGS 中定义
# 默认有: qwen3-8b, gpt-41-mini
LM_NAME="qwen3-8b"                  # 模型名称
LM_MODEL="openai/arbor:qwen/qwen3-8b"  # 模型路径
LM_TEMPERATURE=0.6                  # 采样温度
LM_TOP_P=0.95                       # Top-p 采样
LM_TOP_K=20                         # Top-k 采样

# --- 实验控制 ---
SEED=0                              # 随机种子
NUM_THREADS=32                      # 并行线程数
DRY_RUN=false                       # 是否空运行（测试模式）

# --- GPU 配置 ---
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # 可用GPU（根据你的硬件修改）

# --- 环境变量 ---
export JAX_PLATFORMS=cpu            # JAX使用CPU（避免CuDNN冲突）
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

# --- API密钥（从环境变量读取） ---
# 确保已设置: export OPENAI_API_KEY=sk-xxx
# 确保已设置: export WANDB_API_KEY=xxx
if [ -z "$OPENAI_API_KEY" ]; then
    echo "警告: OPENAI_API_KEY 未设置"
fi
if [ -z "$WANDB_API_KEY" ]; then
    echo "警告: WANDB_API_KEY 未设置"
fi

# ============================================================================
# 【高级配置】可选参数
# ============================================================================

# 是否使用其他优化器的缓存
USE_CACHE_FROM_OPT=""               # 留空表示不使用缓存，或设置为优化器名称如 "Baseline"

# ============================================================================
# 构建 LM 配置 JSON
# ============================================================================
LM_CONFIG=$(cat <<EOF
{
    "name": "$LM_NAME",
    "model": "$LM_MODEL",
    "api_key": "API_KEY",
    "api_base": "http://localhost:{portnum}/v1/",
    "temperature": $LM_TEMPERATURE,
    "top_p": $LM_TOP_P,
    "top_k": $LM_TOP_K,
    "launch_kwargs": {"max_context_length": 8192},
    "train_kwargs": {}
}
EOF
)

# ============================================================================
# 日志和输出
# ============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/exp_${BENCHMARK_NAME}_${OPTIMIZER_NAME}_${LM_NAME}_${TIMESTAMP}.log"

echo "========================================================================"
echo "                    GEPA 实验运行配置"
echo "========================================================================"
echo "基准测试    : $BENCHMARK_NAME (索引: $BENCHMARK_IDX)"
echo "程序        : $PROGRAM_NAME (索引: $PROGRAM_IDX)"
echo "优化器      : $OPTIMIZER_NAME (索引: $OPTIMIZER_IDX)"
echo "语言模型    : $LM_NAME"
echo "模型路径    : $LM_MODEL"
echo "随机种子    : $SEED"
echo "线程数      : $NUM_THREADS"
echo "GPU设备     : $CUDA_VISIBLE_DEVICES"
echo "空运行      : $DRY_RUN"
echo "------------------------------------------------------------------------"
echo "日志文件    : $LOG_FILE"
echo "========================================================================"
echo ""

# ============================================================================
# 构建运行命令
# ============================================================================
# 选择运行方式:
# - 如果设置 USE_UV=true，使用 uv run（需要 Python 3.11）
# - 否则使用当前激活的 Python 环境（推荐）
USE_UV=${USE_UV:-false}

if [ "$USE_UV" = true ]; then
    CMD="uv run python -m scripts.run_experiments"
    echo "使用 uv 运行 (需要 Python 3.11)"
else
    CMD="python -m scripts.run_experiments"
    echo "使用当前 Python 环境: $(which python)"
fi
CMD="$CMD --bm_idx $BENCHMARK_IDX"
CMD="$CMD --benchmark_name \"$BENCHMARK_NAME\""
CMD="$CMD --num_threads $NUM_THREADS"
CMD="$CMD --program_idx $PROGRAM_IDX"
CMD="$CMD --prog_name \"$PROGRAM_NAME\""
CMD="$CMD --opt_idx $OPTIMIZER_IDX"
CMD="$CMD --optim_name \"$OPTIMIZER_NAME\""
CMD="$CMD --lm_config '$LM_CONFIG'"
CMD="$CMD --seed $SEED"

# 添加可选参数
if [ -n "$USE_CACHE_FROM_OPT" ]; then
    CMD="$CMD --use_cache_from_opt $USE_CACHE_FROM_OPT"
fi

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry_run"
fi

# ============================================================================
# 运行实验
# ============================================================================
echo "运行命令:"
echo "$CMD"
echo ""
echo "开始实验..."
echo "========================================================================"
echo ""

# 选择运行模式
read -p "是否后台运行? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 后台运行
    eval "$CMD" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "✓ 实验已在后台启动！"
    echo "  进程 PID: $PID"
    echo "  查看日志: tail -f $LOG_FILE"
    echo "  停止实验: kill $PID"
    echo "$PID" > "$LOG_DIR/last_experiment.pid"
else
    # 前台运行
    eval "$CMD" 2>&1 | tee "$LOG_FILE"
fi

echo ""
echo "实验完成！结果保存在:"
echo "  $(dirname "$PROJECT_ROOT")/experiment_runs_data/"
