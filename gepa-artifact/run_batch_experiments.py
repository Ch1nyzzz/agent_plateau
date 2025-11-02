#!/usr/bin/env python3
"""
GEPA 批量实验运行脚本 (Python版)

用法:
    # 顺序运行所有实验
    python run_batch_experiments.py

    # 并行运行（谨慎使用！）
    python run_batch_experiments.py --parallel

    # 从第 6 个实验开始
    python run_batch_experiments.py --start-from 6

    # 只运行特定实验
    python run_batch_experiments.py --only 1,3,5

    # 跳过某些实验
    python run_batch_experiments.py --skip 2,4

    # 预览要运行的实验（不实际执行）
    python run_batch_experiments.py --dry-run
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import json
import re

# ANSI颜色代码
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

    @staticmethod
    def colored(text: str, color: str) -> str:
        return f"{color}{text}{Colors.NC}"

# 定义所有实验
EXPERIMENTS = [
    # HoverBench 实验
    
    # HotpotQA 实验
    
    {
        "id": 7,
        "benchmark": "HotpotQABench",
        "program": "HotpotMultiHop",
        "optimizer": "GEPA-5",
        "cmd": [
            "python", "-m", "scripts.run_experiments",
            "--bm_idx", "1",
            "--benchmark_name", "HotpotQABench",
            "--num_threads", "4",
            "--program_idx", "0",
            "--prog_name", "HotpotMultiHop",
            "--opt_idx", "1",
            "--optim_name", "GEPA-5",
            "--lm_config", '{"name": "gpt-41-mini", "model": "openai/gpt-4.1-mini-2025-04-14", "api_key": "env:OPENAI_API_KEY", "temperature": 1.0}',
            "--seed", "0",
            "--use_cache_from_opt", "Baseline"
        ]
    },
    {
        "id": 8,
        "benchmark": "HotpotQABench",
        "program": "HotpotMultiHop",
        "optimizer": "GEPA-10",
        "cmd": [
            "python", "-m", "scripts.run_experiments",
            "--bm_idx", "1",
            "--benchmark_name", "HotpotQABench",
            "--num_threads", "4",
            "--program_idx", "0",
            "--prog_name", "HotpotMultiHop",
            "--opt_idx", "2",
            "--optim_name", "GEPA-10",
            "--lm_config", '{"name": "gpt-41-mini", "model": "openai/gpt-4.1-mini-2025-04-14", "api_key": "env:OPENAI_API_KEY", "temperature": 1.0}',
            "--seed", "0",
            "--use_cache_from_opt", "Baseline"
        ]
    },
    {
        "id": 9,
        "benchmark": "HotpotQABench",
        "program": "HotpotMultiHop",
        "optimizer": "GEPA-15",
        "cmd": [
            "python", "-m", "scripts.run_experiments",
            "--bm_idx", "1",
            "--benchmark_name", "HotpotQABench",
            "--num_threads", "4",
            "--program_idx", "0",
            "--prog_name", "HotpotMultiHop",
            "--opt_idx", "3",
            "--optim_name", "GEPA-15",
            "--lm_config", '{"name": "gpt-41-mini", "model": "openai/gpt-4.1-mini-2025-04-14", "api_key": "env:OPENAI_API_KEY", "temperature": 1.0}',
            "--seed", "0",
            "--use_cache_from_opt", "Baseline"
        ]
    },
    {
        "id": 10,
        "benchmark": "HotpotQABench",
        "program": "HotpotMultiHop",
        "optimizer": "GEPA-20",
        "cmd": [
            "python", "-m", "scripts.run_experiments",
            "--bm_idx", "1",
            "--benchmark_name", "HotpotQABench",
            "--num_threads", "4",
            "--program_idx", "0",
            "--prog_name", "HotpotMultiHop",
            "--opt_idx", "4",
            "--optim_name", "GEPA-20",
            "--lm_config", '{"name": "gpt-41-mini", "model": "openai/gpt-4.1-mini-2025-04-14", "api_key": "env:OPENAI_API_KEY", "temperature": 1.0}',
            "--seed", "0",
            "--use_cache_from_opt", "Baseline"
        ]
    },
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="GEPA 批量实验运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--parallel", action="store_true", help="并行运行实验（谨慎使用）")
    parser.add_argument("--start-from", type=int, default=1, help="从第 N 个实验开始")
    parser.add_argument("--only", type=str, help="只运行指定的实验 (逗号分隔，如: 1,3,5)")
    parser.add_argument("--skip", type=str, help="跳过指定的实验 (逗号分隔，如: 2,4)")
    parser.add_argument("--dry-run", action="store_true", help="预览要运行的实验，不实际执行")
    return parser.parse_args()

def filter_experiments(experiments: List[Dict], args) -> List[Dict]:
    """根据参数过滤实验"""
    filtered = experiments.copy()

    # 处理 --only 参数
    if args.only:
        only_ids = set(int(x.strip()) for x in args.only.split(','))
        filtered = [e for e in filtered if e['id'] in only_ids]

    # 处理 --skip 参数
    if args.skip:
        skip_ids = set(int(x.strip()) for x in args.skip.split(','))
        filtered = [e for e in filtered if e['id'] not in skip_ids]

    # 处理 --start-from 参数
    filtered = [e for e in filtered if e['id'] >= args.start_from]

    return filtered

def run_experiment(exp: Dict, log_dir: Path, timestamp: str) -> Dict:
    """运行单个实验"""
    exp_id = exp['id']
    benchmark = exp['benchmark']
    program = exp['program']
    optimizer = exp['optimizer']
    cmd = exp['cmd']

    exp_log = log_dir / f"exp_{exp_id}_{benchmark}_{program}_{optimizer}_{timestamp}.log"

    print(f"\n{Colors.colored(f'[实验 {exp_id}/{len(EXPERIMENTS)}]', Colors.BLUE)} "
          f"{Colors.colored(benchmark, Colors.GREEN)} - "
          f"{Colors.colored(program, Colors.YELLOW)} - "
          f"{Colors.colored(optimizer, Colors.YELLOW)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    try:
        with open(exp_log, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"{Colors.colored('✓ 完成', Colors.GREEN)} (耗时: {duration:.1f}s)")
            return {
                'id': exp_id,
                'status': 'success',
                'duration': duration,
                'log': exp_log
            }
        else:
            print(f"{Colors.colored('✗ 失败', Colors.RED)} (耗时: {duration:.1f}s)")
            print(f"{Colors.colored(f'查看日志: {exp_log}', Colors.YELLOW)}")
            return {
                'id': exp_id,
                'status': 'failed',
                'duration': duration,
                'log': exp_log
            }

    except Exception as e:
        duration = time.time() - start_time
        print(f"{Colors.colored(f'✗ 异常: {str(e)}', Colors.RED)} (耗时: {duration:.1f}s)")
        return {
            'id': exp_id,
            'status': 'error',
            'duration': duration,
            'log': exp_log,
            'error': str(e)
        }

def extract_score_from_log(log_file: Path) -> str:
    """从日志文件中提取分数"""
    if not log_file.exists():
        return "日志文件不存在"

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # 查找 "Average Metric:" 行
        match = re.search(r'Average Metric:.*', content)
        if match:
            return match.group(0)
        else:
            return "未找到分数信息"
    except Exception as e:
        return f"读取失败: {str(e)}"

def generate_summary(results: List[Dict], experiments: List[Dict], log_dir: Path, timestamp: str):
    """生成实验结果摘要"""
    summary_file = log_dir / f"experiment_summary_{timestamp}.txt"

    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("实验结果摘要\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        # 按 benchmark 分组
        hover_results = [r for r in results if any(e['id'] == r['id'] and e['benchmark'] == 'hoverBench' for e in experiments)]
        hotpot_results = [r for r in results if any(e['id'] == r['id'] and e['benchmark'] == 'HotpotQABench' for e in experiments)]

        for group_name, group_results in [("HoverBench", hover_results), ("HotpotQABench", hotpot_results)]:
            if group_results:
                f.write(f"\n{group_name}\n")
                f.write("-" * 60 + "\n")

                for result in sorted(group_results, key=lambda x: x['id']):
                    exp = next(e for e in experiments if e['id'] == result['id'])
                    f.write(f"\n[{exp['id']}] {exp['optimizer']}\n")
                    f.write(f"  状态: {result['status']}\n")
                    f.write(f"  耗时: {result['duration']:.1f}s\n")

                    if result['status'] == 'success' and result['log'].exists():
                        score = extract_score_from_log(result['log'])
                        f.write(f"  结果: {score}\n")

                    f.write(f"  日志: {result['log']}\n")

        # 统计信息
        f.write("\n" + "=" * 60 + "\n")
        f.write("统计信息\n")
        f.write("=" * 60 + "\n")
        total = len(results)
        success = len([r for r in results if r['status'] == 'success'])
        failed = len([r for r in results if r['status'] == 'failed'])
        error = len([r for r in results if r['status'] == 'error'])
        total_time = sum(r['duration'] for r in results)

        f.write(f"总实验数: {total}\n")
        f.write(f"成功: {success}\n")
        f.write(f"失败: {failed}\n")
        f.write(f"错误: {error}\n")
        f.write(f"总耗时: {total_time/3600:.2f}h\n")

    print(f"\n{Colors.colored('摘要已保存到:', Colors.GREEN)} {summary_file}")
    return summary_file

def main():
    args = parse_args()

    # 创建日志目录
    log_dir = Path("batch_experiment_logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 过滤实验
    experiments_to_run = filter_experiments(EXPERIMENTS, args)

    if not experiments_to_run:
        print(f"{Colors.colored('错误: 没有要运行的实验', Colors.RED)}")
        return

    # 打印标题
    print(f"{Colors.colored('=' * 60, Colors.BLUE)}")
    print(f"{Colors.colored('  GEPA 批量实验运行脚本 (Python版)', Colors.BLUE)}")
    print(f"{Colors.colored('=' * 60, Colors.BLUE)}")
    print(f"总实验数: {Colors.colored(str(len(experiments_to_run)), Colors.YELLOW)}")
    print(f"并行模式: {Colors.colored('开启' if args.parallel else '关闭', Colors.YELLOW)}")
    print(f"日志目录: {Colors.colored(str(log_dir), Colors.YELLOW)}")

    # Dry run 模式
    if args.dry_run:
        print(f"\n{Colors.colored('预览模式 (不会实际执行)', Colors.YELLOW)}\n")
        for exp in experiments_to_run:
            print(f"[{exp['id']}] {exp['benchmark']} - {exp['program']} - {exp['optimizer']}")
        return

    print(f"{Colors.colored('=' * 60, Colors.BLUE)}\n")

    # 运行实验
    script_start_time = time.time()
    results = []

    if args.parallel:
        print(f"{Colors.colored('警告: 并行模式会同时运行多个实验！', Colors.YELLOW)}")
        print(f"{Colors.colored('按 Ctrl+C 取消，或等待 5 秒后开始...', Colors.YELLOW)}")
        time.sleep(5)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_experiment, exp, log_dir, timestamp) for exp in experiments_to_run]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
    else:
        for exp in experiments_to_run:
            result = run_experiment(exp, log_dir, timestamp)
            results.append(result)

            # 如果失败，询问是否继续
            if result['status'] != 'success':
                response = input(f"{Colors.colored('是否继续运行剩余实验？ (y/n): ', Colors.YELLOW)}")
                if response.lower() not in ['y', 'yes']:
                    print("用户选择终止")
                    break

    # 计算总耗时
    total_duration = time.time() - script_start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    print(f"\n{Colors.colored('=' * 60, Colors.BLUE)}")
    print(f"{Colors.colored('所有实验完成！', Colors.GREEN)}")
    print(f"{Colors.colored('=' * 60, Colors.BLUE)}")
    print(f"总耗时: {Colors.colored(f'{hours}h {minutes}m {seconds}s', Colors.YELLOW)}")

    # 生成摘要
    summary_file = generate_summary(results, experiments_to_run, log_dir, timestamp)

    # 打印摘要
    with open(summary_file, 'r') as f:
        print(f"\n{f.read()}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.colored('用户中断', Colors.YELLOW)}")
        sys.exit(1)
