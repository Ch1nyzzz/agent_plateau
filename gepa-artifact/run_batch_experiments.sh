#!/bin/bash

# GEPA 批量实验运行脚本
# 用法: bash run_batch_experiments.sh [--parallel] [--start-from N]

set -e  # 遇到错误时退出（可选，根据需要调整）

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志目录
LOG_DIR="./batch_experiment_logs"
mkdir -p "$LOG_DIR"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/batch_run_${TIMESTAMP}.log"

# 参数解析
PARALLEL=false
START_FROM=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=true
            shift
            ;;
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--parallel] [--start-from N]"
            exit 1
            ;;
    esac
done

# 定义所有实验命令
declare -a EXPERIMENTS=(
    # HoverBench 实验
    "1|hoverBench|HoverMultiHop|GEPA-10|python -m scripts.run_experiments --bm_idx 0 --benchmark_name 'hoverBench' --num_threads 4 --program_idx 0 --prog_name 'HoverMultiHop' --opt_idx 2 --optim_name 'GEPA-10' --lm_config '{\"name\": \"gpt-41-mini\", \"model\": \"openai/gpt-4.1-mini-2025-04-14\", \"api_key\": \"env:OPENAI_API_KEY\", \"temperature\": 1.0}' --seed 0 --use_cache_from_opt Baseline"

    "2|hoverBench|HoverMultiHop|GEPA-15|python -m scripts.run_experiments --bm_idx 0 --benchmark_name 'hoverBench' --num_threads 4 --program_idx 0 --prog_name 'HoverMultiHop' --opt_idx 3 --optim_name 'GEPA-15' --lm_config '{\"name\": \"gpt-41-mini\", \"model\": \"openai/gpt-4.1-mini-2025-04-14\", \"api_key\": \"env:OPENAI_API_KEY\", \"temperature\": 1.0}' --seed 0 --use_cache_from_opt Baseline"

    "3|hoverBench|HoverMultiHop|GEPA-20|python -m scripts.run_experiments --bm_idx 0 --benchmark_name 'hoverBench' --num_threads 4 --program_idx 0 --prog_name 'HoverMultiHop' --opt_idx 4 --optim_name 'GEPA-20' --lm_config '{\"name\": \"gpt-41-mini\", \"model\": \"openai/gpt-4.1-mini-2025-04-14\", \"api_key\": \"env:OPENAI_API_KEY\", \"temperature\": 1.0}' --seed 0 --use_cache_from_opt Baseline"

    "4|hoverBench|HoverMultiHop|GEPA-25|python -m scripts.run_experiments --bm_idx 0 --benchmark_name 'hoverBench' --num_threads 4 --program_idx 0 --prog_name 'HoverMultiHop' --opt_idx 5 --optim_name 'GEPA-25' --lm_config '{\"name\": \"gpt-41-mini\", \"model\": \"openai/gpt-4.1-mini-2025-04-14\", \"api_key\": \"env:OPENAI_API_KEY\", \"temperature\": 1.0}' --seed 0 --use_cache_from_opt Baseline"

    "5|hoverBench|HoverMultiHop|GEPA-50|python -m scripts.run_experiments --bm_idx 0 --benchmark_name 'hoverBench' --num_threads 4 --program_idx 0 --prog_name 'HoverMultiHop' --opt_idx 6 --optim_name 'GEPA-50' --lm_config '{\"name\": \"gpt-41-mini\", \"model\": \"openai/gpt-4.1-mini-2025-04-14\", \"api_key\": \"env:OPENAI_API_KEY\", \"temperature\": 1.0}' --seed 0 --use_cache_from_opt Baseline"

    # HotpotQA 实验
    "6|HotpotQABench|HotpotMultiHop|Baseline|python -m scripts.run_experiments --bm_idx 1 --benchmark_name 'HotpotQABench' --num_threads 4 --program_idx 0 --prog_name 'HotpotMultiHop' --opt_idx 0 --optim_name 'Baseline' --lm_config '{\"name\": \"gpt-41-mini\", \"model\": \"openai/gpt-4.1-mini-2025-04-14\", \"api_key\": \"env:OPENAI_API_KEY\", \"temperature\": 1.0}' --seed 0"

    "7|HotpotQABench|HotpotMultiHop|GEPA-5|python -m scripts.run_experiments --bm_idx 1 --benchmark_name 'HotpotQABench' --num_threads 4 --program_idx 0 --prog_name 'HotpotMultiHop' --opt_idx 1 --optim_name 'GEPA-5' --lm_config '{\"name\": \"gpt-41-mini\", \"model\": \"openai/gpt-4.1-mini-2025-04-14\", \"api_key\": \"env:OPENAI_API_KEY\", \"temperature\": 1.0}' --seed 0 --use_cache_from_opt Baseline"

    "8|HotpotQABench|HotpotMultiHop|GEPA-10|python -m scripts.run_experiments --bm_idx 1 --benchmark_name 'HotpotQABench' --num_threads 4 --program_idx 0 --prog_name 'HotpotMultiHop' --opt_idx 2 --optim_name 'GEPA-10' --lm_config '{\"name\": \"gpt-41-mini\", \"model\": \"openai/gpt-4.1-mini-2025-04-14\", \"api_key\": \"env:OPENAI_API_KEY\", \"temperature\": 1.0}' --seed 0 --use_cache_from_opt Baseline"

    "9|HotpotQABench|HotpotMultiHop|GEPA-15|python -m scripts.run_experiments --bm_idx 1 --benchmark_name 'HotpotQABench' --num_threads 4 --program_idx 0 --prog_name 'HotpotMultiHop' --opt_idx 3 --optim_name 'GEPA-15' --lm_config '{\"name\": \"gpt-41-mini\", \"model\": \"openai/gpt-4.1-mini-2025-04-14\", \"api_key\": \"env:OPENAI_API_KEY\", \"temperature\": 1.0}' --seed 0 --use_cache_from_opt Baseline"

    "10|HotpotQABench|HotpotMultiHop|GEPA-20|python -m scripts.run_experiments --bm_idx 1 --benchmark_name 'HotpotQABench' --num_threads 4 --program_idx 0 --prog_name 'HotpotMultiHop' --opt_idx 4 --optim_name 'GEPA-20' --lm_config '{\"name\": \"gpt-41-mini\", \"model\": \"openai/gpt-4.1-mini-2025-04-14\", \"api_key\": \"env:OPENAI_API_KEY\", \"temperature\": 1.0}' --seed 0 --use_cache_from_opt Baseline"

    "11|HotpotQABench|HotpotMultiHop|GEPA-25|python -m scripts.run_experiments --bm_idx 1 --benchmark_name 'HotpotQABench' --num_threads 4 --program_idx 0 --prog_name 'HotpotMultiHop' --opt_idx 5 --optim_name 'GEPA-25' --lm_config '{\"name\": \"gpt-41-mini\", \"model\": \"openai/gpt-4.1-mini-2025-04-14\", \"api_key\": \"env:OPENAI_API_KEY\", \"temperature\": 1.0}' --seed 0 --use_cache_from_opt Baseline"

    "12|HotpotQABench|HotpotMultiHop|GEPA-50|python -m scripts.run_experiments --bm_idx 1 --benchmark_name 'HotpotQABench' --num_threads 4 --program_idx 0 --prog_name 'HotpotMultiHop' --opt_idx 6 --optim_name 'GEPA-50' --lm_config '{\"name\": \"gpt-41-mini\", \"model\": \"openai/gpt-4.1-mini-2025-04-14\", \"api_key\": \"env:OPENAI_API_KEY\", \"temperature\": 1.0}' --seed 0 --use_cache_from_opt Baseline"
)

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}

# 记录开始时间
SCRIPT_START_TIME=$(date +%s)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GEPA 批量实验运行脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "总实验数: ${YELLOW}$TOTAL_EXPERIMENTS${NC}"
echo -e "从第 ${YELLOW}${START_FROM}${NC} 个实验开始"
echo -e "并行模式: ${YELLOW}$([ "$PARALLEL" = true ] && echo "开启" || echo "关闭")${NC}"
echo -e "日志文件: ${YELLOW}$MAIN_LOG${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 记录到日志
{
    echo "========================================="
    echo "批量实验运行日志"
    echo "开始时间: $(date)"
    echo "========================================="
} >> "$MAIN_LOG"

# 运行单个实验的函数
run_experiment() {
    local exp_info=$1

    IFS='|' read -r idx benchmark program optimizer command <<< "$exp_info"

    local exp_log="$LOG_DIR/exp_${idx}_${benchmark}_${program}_${optimizer}_${TIMESTAMP}.log"
    local start_time=$(date +%s)

    echo -e "\n${BLUE}[实验 $idx/$TOTAL_EXPERIMENTS]${NC} ${GREEN}$benchmark${NC} - ${YELLOW}$program${NC} - ${YELLOW}$optimizer${NC}"
    echo -e "开始时间: $(date)"

    # 记录到主日志
    {
        echo ""
        echo "========================================="
        echo "[实验 $idx/$TOTAL_EXPERIMENTS] $benchmark - $program - $optimizer"
        echo "开始时间: $(date)"
        echo "命令: $command"
        echo "========================================="
    } >> "$MAIN_LOG"

    # 运行实验
    if eval "$command" > "$exp_log" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${GREEN}✓ 完成${NC} (耗时: ${duration}s)"
        echo "[实验 $idx] 成功 (耗时: ${duration}s)" >> "$MAIN_LOG"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${RED}✗ 失败${NC} (耗时: ${duration}s)"
        echo -e "${YELLOW}查看日志: $exp_log${NC}"
        echo "[实验 $idx] 失败 (耗时: ${duration}s)" >> "$MAIN_LOG"

        # 如果不是并行模式，询问是否继续
        if [ "$PARALLEL" = false ]; then
            echo -e "${YELLOW}是否继续运行剩余实验？ (y/n)${NC}"
            read -r continue_choice
            if [[ ! "$continue_choice" =~ ^[Yy]$ ]]; then
                echo "用户选择终止"
                exit 1
            fi
        fi
    fi
}

# 如果是并行模式
if [ "$PARALLEL" = true ]; then
    echo -e "${YELLOW}警告: 并行模式会同时运行所有实验，请确保有足够的计算资源！${NC}"
    echo -e "${YELLOW}按 Ctrl+C 取消，或等待 5 秒后开始...${NC}"
    sleep 5

    for exp_info in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r idx benchmark program optimizer command <<< "$exp_info"

        if [ "$idx" -ge "$START_FROM" ]; then
            run_experiment "$exp_info" &
        fi
    done

    # 等待所有后台任务完成
    wait

else
    # 顺序模式
    for exp_info in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r idx benchmark program optimizer command <<< "$exp_info"

        if [ "$idx" -ge "$START_FROM" ]; then
            run_experiment "$exp_info"
        else
            echo -e "${YELLOW}[实验 $idx/$TOTAL_EXPERIMENTS] 跳过${NC}"
        fi
    done
fi

# 计算总耗时
SCRIPT_END_TIME=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}所有实验完成！${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "总耗时: ${YELLOW}${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
echo -e "主日志: ${YELLOW}$MAIN_LOG${NC}"
echo -e "实验日志目录: ${YELLOW}$LOG_DIR${NC}"
echo -e "${BLUE}========================================${NC}"

# 记录到日志
{
    echo ""
    echo "========================================="
    echo "所有实验完成"
    echo "结束时间: $(date)"
    echo "总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "========================================="
} >> "$MAIN_LOG"

# 生成实验结果摘要
echo -e "\n${BLUE}生成实验结果摘要...${NC}"
SUMMARY_FILE="$LOG_DIR/experiment_summary_${TIMESTAMP}.txt"

{
    echo "========================================="
    echo "实验结果摘要"
    echo "生成时间: $(date)"
    echo "========================================="
    echo ""

    for exp_info in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r idx benchmark program optimizer command <<< "$exp_info"

        if [ "$idx" -ge "$START_FROM" ]; then
            exp_log="$LOG_DIR/exp_${idx}_${benchmark}_${program}_${optimizer}_${TIMESTAMP}.log"

            echo "[$idx] $benchmark - $program - $optimizer"

            if [ -f "$exp_log" ]; then
                # 尝试提取最终分数
                if grep -q "Average Metric:" "$exp_log"; then
                    score=$(grep "Average Metric:" "$exp_log" | tail -1)
                    echo "  结果: $score"
                else
                    echo "  状态: 检查日志文件"
                fi
                echo "  日志: $exp_log"
            else
                echo "  状态: 未运行或日志缺失"
            fi
            echo ""
        fi
    done
} > "$SUMMARY_FILE"

echo -e "${GREEN}摘要已保存到: $SUMMARY_FILE${NC}"
cat "$SUMMARY_FILE"
