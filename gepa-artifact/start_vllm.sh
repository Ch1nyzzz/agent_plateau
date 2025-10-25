#!/bin/bash

# vLLM 本地推理服务启动脚本
# 使用方法: bash start_vllm.sh

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# 检查 .env 文件是否存在
if [ ! -f "$ENV_FILE" ]; then
    print_error ".env 文件不存在！"
    print_info "请复制 .env.template 为 .env 并填入实际配置："
    echo "  cp .env.template .env"
    exit 1
fi

# 加载环境变量
print_info "正在加载环境变量..."
export $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs)

# 设置默认值
VLLM_MODEL_PATH=${VLLM_MODEL_PATH:-"/home/yuhan/model_zoo/Qwen3-8B"}
VLLM_HOST=${VLLM_HOST:-"localhost"}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-15000}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-256}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}

# 检查模型路径是否存在
if [ ! -d "$VLLM_MODEL_PATH" ]; then
    print_error "模型路径不存在: $VLLM_MODEL_PATH"
    print_info "请检查 .env 文件中的 VLLM_MODEL_PATH 配置"
    exit 1
fi

# 检查端口是否被占用
if lsof -Pi :$VLLM_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    print_warn "端口 $VLLM_PORT 已被占用"
    read -p "是否要杀死占用该端口的进程? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "正在杀死占用端口 $VLLM_PORT 的进程..."
        lsof -ti:$VLLM_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    else
        print_error "请更改端口或手动关闭占用该端口的进程"
        exit 1
    fi
fi

# 检查是否安装了 vllm
if ! command -v vllm &> /dev/null && ! python -c "import vllm" &> /dev/null; then
    print_error "vLLM 未安装！"
    print_info "请先安装 vLLM:"
    echo "  pip install vllm"
    exit 1
fi

# 显示配置信息
print_info "vLLM 配置信息:"
echo "  模型路径: $VLLM_MODEL_PATH"
echo "  监听地址: $VLLM_HOST:$VLLM_PORT"
echo "  GPU 内存利用率: $VLLM_GPU_MEMORY_UTILIZATION"
echo "  Tensor 并行度: $VLLM_TENSOR_PARALLEL_SIZE"
echo "  最大模型长度: $VLLM_MAX_MODEL_LEN"
echo "  最大并发序列数: $VLLM_MAX_NUM_SEQS"
echo "  最大批次Token数: $VLLM_MAX_NUM_BATCHED_TOKENS"
echo ""

# 启动 vLLM 服务器
print_info "正在启动 vLLM 服务器..."

# 构建启动命令
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model $VLLM_MODEL_PATH \
    --host $VLLM_HOST \
    --port $VLLM_PORT \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $VLLM_TENSOR_PARALLEL_SIZE \
    --max-model-len $VLLM_MAX_MODEL_LEN \
    --max-num-seqs $VLLM_MAX_NUM_SEQS \
    --max-num-batched-tokens $VLLM_MAX_NUM_BATCHED_TOKENS \
    --trust-remote-code"

# 显示启动命令
print_info "启动命令:"
echo "$VLLM_CMD"
echo ""

# 执行启动命令
print_info "服务器启动中，请稍候..."
print_info "API 地址: http://$VLLM_HOST:$VLLM_PORT/v1"
print_info "按 Ctrl+C 可以停止服务"
echo ""

# 执行命令
eval $VLLM_CMD
