"""
GEPA Experiment Script - Optimized Ray + vLLM Multi-GPU Inference
使用 Ray 实现真正的分布式并行推理，针对 4 卡 GPU 优化

Features:
1. Ray-based distributed inference (vLLMOfflineHybridParallel)
2. GEPA optimizer training
3. Benchmark evaluation
"""

import os
import sys
import time
import argparse
from pathlib import Path

import dspy

# Import optimized Ray + vLLM adapter
from gepa_artifact.utils.vllm_dspy_adapter import vLLMOfflineHybridParallel

# Import benchmark modules
from gepa_artifact.benchmarks.AIME import benchmark as aime_metas
from gepa_artifact.benchmarks.livebench_math import benchmark as lb_math_metas
from gepa_artifact.benchmarks.hover import benchmark as hover_metas
from gepa_artifact.benchmarks.papillon import benchmark as papillon_metas
from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpotqa_metas
from gepa_artifact.benchmarks.IFBench import benchmark as ifbench_metas

# Import GEPA optimizer
from gepa_artifact.gepa.gepa import GEPA
from gepa_artifact.utils.capture_stream_logger import Logger


def setup_lm(args):
    """
    Setup language model using Ray + vLLM distributed inference
    使用 Ray 进行分布式推理，支持张量并行和数据并行的混合
    """
    print("=" * 80)
    print("1. Initialize Language Model (Ray + vLLM)")
    print("=" * 80)

    # 计算总 GPU 数量
    total_gpus = args.tensor_parallel_size * args.num_model_instances

    print(f"配置信息:")
    print(f"  模型路径: {args.model_path}")
    print(f"  Tensor Parallel Size: {args.tensor_parallel_size} (每个模型实例使用 {args.tensor_parallel_size} 张 GPU)")
    print(f"  Model Instances: {args.num_model_instances} (数据并行，运行 {args.num_model_instances} 个模型副本)")
    print(f"  总 GPU 使用: {total_gpus} 张")
    print(f"  预期吞吐量提升: ~{args.num_model_instances}x")

    # 初始化 Ray + vLLM 混合并行推理
    lm = vLLMOfflineHybridParallel(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        num_model_instances=args.num_model_instances,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )

    # Configure DSPy
    dspy.configure(lm=lm)

    print("\n✓ Language model initialized successfully")
    return lm


def load_benchmark(args):
    """
    Load benchmark dataset
    """
    print("\n" + "=" * 80)
    print("2. Load Benchmark Dataset")
    print("=" * 80)
    print(f"Benchmark: {args.benchmark}")

    if args.benchmark == "aime":
        cur_meta = aime_metas
    elif args.benchmark == "lb_math":
        cur_meta = lb_math_metas
    elif args.benchmark == "hover":
        cur_meta = hover_metas
    elif args.benchmark == "papillon":
        cur_meta = papillon_metas
    elif args.benchmark == "hotpotqa":
        cur_meta = hotpotqa_metas
    elif args.benchmark == "ifbench":
        cur_meta = ifbench_metas
    else:
        raise ValueError(f"Invalid benchmark: {args.benchmark}")

    # Initialize the benchmark
    bench = cur_meta[0].benchmark()

    print(f"Train set size: {len(bench.train_set)}")
    print(f"Val set size: {len(bench.val_set)}")
    print(f"Test set size: {len(bench.test_set)}")

    # Debug mode: use subset of data
    if args.debug:
        print("\n[DEBUG MODE] Using first 10 samples only")
        bench.train_set = bench.train_set[:10]
        bench.val_set = bench.val_set[:5]
        bench.test_set = bench.test_set[:5]

    return cur_meta, bench


def load_program(cur_meta):
    """
    Load DSPy program
    """
    print("\n" + "=" * 80)
    print("3. Load DSPy Program")
    print("=" * 80)

    program = cur_meta[0].program[0]
    print(f"Program type: {type(program).__name__}")
    print(f"Program structure:")
    print(program)

    return program


def evaluate_program(program, cur_meta, bench, args, stage="base"):
    """
    Evaluate program performance

    Args:
        program: DSPy program
        cur_meta: benchmark metadata
        bench: benchmark dataset
        args: command line arguments
        stage: evaluation stage identifier ("base" or "optimized")
    """
    print("\n" + "=" * 80)
    print(f"4. Evaluate {'Base' if stage == 'base' else 'Optimized'} Program")
    print("=" * 80)

    # Configure rollouts if specified
    if args.num_rollouts > 1:
        print(f"\nConfiguring {args.num_rollouts} rollouts per prompt")
        # Configure the language model to generate multiple outputs
        # This is done by setting the 'n' parameter in the LM kwargs
        import dspy
        lm = dspy.settings.lm
        if lm is not None:
            # Store original config
            original_kwargs = getattr(lm, 'kwargs', {})
            # Update with rollouts config
            if hasattr(lm, 'kwargs'):
                lm.kwargs = {**original_kwargs, 'n': args.num_rollouts}

            print(f"  LM configured for {args.num_rollouts} rollouts per prompt")

    # Create evaluator
    evaluate = dspy.Evaluate(
        devset=bench.test_set,
        metric=cur_meta[0].metric,
        num_threads=args.num_eval_threads,
        display_table=True,
        display_progress=True,
        max_errors=100 * len(bench.test_set)
    )

    print(f"\nStarting evaluation ({len(bench.test_set)} samples)...")
    if args.num_rollouts > 1:
        print(f"  Generating {args.num_rollouts} rollouts per sample (total: {len(bench.test_set) * args.num_rollouts} generations)")

    start_time = time.time()
    score = evaluate(program)
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Evaluation Complete - {stage.upper()}")
    print(f"{'='*80}")
    print(f"Score: {score:.2%}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Avg per sample: {elapsed/len(bench.test_set):.2f}s")
    if args.num_rollouts > 1:
        print(f"Avg per generation: {elapsed/(len(bench.test_set) * args.num_rollouts):.2f}s")

    return score


def optimize_program(program, cur_meta, bench, args):
    """
    Optimize program using GEPA optimizer
    """
    print("\n" + "=" * 80)
    print("5. Optimize Program with GEPA")
    print("=" * 80)

    # Create run directory
    runs_dir = Path(args.output_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")
    runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {runs_dir}")

    # Create logger
    gepa_logger = Logger(str(runs_dir / "run_log.txt"))

    # Prepare feedback function
    if cur_meta[0].feedback_fn_maps is None or cur_meta[0].feedback_fn_maps[0] is None:
        def feedback_func(predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace):
            pred = cur_meta[0].metric_with_feedback(module_inputs, module_outputs, None)
            return {
                "feedback_score": pred.score,
                "feedback_text": pred.feedback,
            }

        feedback_fn_map = {k: feedback_func for k, v in program.named_predictors()}
    else:
        feedback_fn_map = cur_meta[0].feedback_fn_maps[0]

    # Create GEPA optimizer
    print("\nInitializing GEPA optimizer...")
    optimizer = GEPA(
        named_predictor_to_feedback_fn_map=feedback_fn_map,
        knowledgebase_qe=None,
        metric=cur_meta[0].metric,
        run_linearized_gepa=False,
        use_merge=args.use_merge,
        set_for_merge_minibatch='val',
        track_scores_on='val',
        max_metric_calls=args.max_metric_calls,
        run_dir=str(runs_dir),
        logger=gepa_logger,
        num_threads=args.num_optimize_threads
    )

    # Prepare train and validation sets
    trainset = bench.train_set
    valset = bench.val_set[:len(bench.val_set)//2] if not args.debug else bench.val_set

    print(f"\nOptimization config:")
    print(f"  Train set size: {len(trainset)}")
    print(f"  Val set size: {len(valset)}")
    print(f"  Max metric calls: {args.max_metric_calls}")
    print(f"  Use merge: {args.use_merge}")
    print(f"  Optimize threads: {args.num_optimize_threads}")

    # Execute optimization
    print("\nStarting optimization...")
    start_time = time.time()
    optimized_program = optimizer.compile(
        program,
        trainset=trainset,
        valset=valset,
    )
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Optimization Complete")
    print(f"{'='*80}")
    print(f"Time: {elapsed:.2f}s ({elapsed/60:.2f}min)")
    print(f"Run directory: {runs_dir}")

    return optimized_program


def print_optimized_prompts(optimized_program):
    """
    Print optimized prompts
    """
    print("\n" + "=" * 80)
    print("6. Optimized Prompts")
    print("=" * 80)

    for name, pred in optimized_program.named_predictors():
        print("\n" + "=" * 80)
        print(f"Predictor: {name}")
        print("=" * 80)
        print("\nPrompt:")
        print(pred.signature.instructions)
        print("\n" + "*" * 80)


def main():
    """
    Main function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="GEPA Experiment Script - Optimized Ray + vLLM Multi-GPU Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model-path",
        type=str,
        default="/home/yuhan/model_zoo/Qwen3-8B",
        help="Path to model"
    )
    model_group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (number of GPUs per model instance). "
             "对于 8B 模型，建议设为 1（单卡足够）；对于 70B+ 模型，可以设为 2 或 4"
    )
    model_group.add_argument(
        "--num-model-instances",
        type=int,
        default=4,
        help="Number of model instances (data parallel). "
             "Total GPUs = tensor_parallel_size × num_model_instances. "
             "对于 4 张 GPU，建议 tensor_parallel_size=1, num_model_instances=4 以最大化吞吐量"
    )
    model_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (0.0-1.0)"
    )

    # Inference parameters
    inference_group = parser.add_argument_group('Inference Parameters')
    inference_group.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    inference_group.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum number of tokens to generate"
    )
    inference_group.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model length"
    )
    inference_group.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter"
    )
    inference_group.add_argument(
        "--max-num-seqs",
        type=int,
        default=128,
        help="Maximum number of sequences (requests) to process in parallel in vLLM"
    )
    inference_group.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of rollouts (candidate generations) per prompt during evaluation. "
             "Higher values enable sampling multiple outputs for each input. "
             "Note: This is different from num_dspy_examples_per_gepa_step which controls batch size during optimization."
    )

    # Experiment configuration
    experiment_group = parser.add_argument_group('Experiment Configuration')
    experiment_group.add_argument(
        "--benchmark",
        type=str,
        default="aime",
        choices=["aime", "lb_math", "hover", "papillon", "hotpotqa", "ifbench"],
        help="Benchmark to use"
    )
    experiment_group.add_argument(
        "--skip-base-eval",
        action="store_true",
        help="Skip base program evaluation"
    )
    experiment_group.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Skip optimization step"
    )
    experiment_group.add_argument(
        "--skip-optimized-eval",
        action="store_true",
        help="Skip optimized program evaluation"
    )
    experiment_group.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (use less data)"
    )

    # GEPA optimizer configuration
    gepa_group = parser.add_argument_group('GEPA Optimizer Configuration')
    gepa_group.add_argument(
        "--use-merge",
        action="store_true",
        default=True,
        help="Use merge strategy"
    )
    gepa_group.add_argument(
        "--max-metric-calls",
        type=int,
        default=700,
        help="Maximum number of metric calls"
    )
    gepa_group.add_argument(
        "--num-optimize-threads",
        type=int,
        default=40,
        help="Number of threads for optimization"
    )

    # Evaluation configuration
    eval_group = parser.add_argument_group('Evaluation Configuration')
    eval_group.add_argument(
        "--num-eval-threads",
        type=int,
        default=32,
        help="Number of threads for evaluation"
    )

    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="./runs",
        help="Output directory"
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("GEPA Experiment Script - Ray + vLLM Optimized")
    print("=" * 80)
    print(f"\n实验配置:")
    print(f"  模型路径: {args.model_path}")
    print(f"  推理模式: Ray + vLLM 混合并行")
    print(f"  Tensor Parallel: {args.tensor_parallel_size} (每个实例的 GPU 数)")
    print(f"  Data Parallel: {args.num_model_instances} (模型实例数)")
    print(f"  总 GPU 数量: {args.tensor_parallel_size * args.num_model_instances}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Debug mode: {args.debug}")
    print()

    # Set environment variable if needed
    if not os.environ.get('OPENAI_API_KEY'):
        # Some DSPy features may need this, set a placeholder
        os.environ['OPENAI_API_KEY'] = "EMPTY"

    # Execute experiment workflow
    try:
        # 1. Initialize language model
        lm = setup_lm(args)

        # 2. Load benchmark
        cur_meta, bench = load_benchmark(args)

        # 3. Load program
        program = load_program(cur_meta)

        # 4. Evaluate base program (optional)
        base_score = None
        if not args.skip_base_eval:
            base_score = evaluate_program(program, cur_meta, bench, args, stage="base")

        # 5. Optimize program (optional)
        optimized_program = None
        if not args.skip_optimize:
            optimized_program = optimize_program(program, cur_meta, bench, args)

        # 6. Evaluate optimized program (optional)
        optimized_score = None
        if optimized_program is not None and not args.skip_optimized_eval:
            optimized_score = evaluate_program(
                optimized_program, cur_meta, bench, args, stage="optimized"
            )

        # 7. Print optimized prompts (if available)
        if optimized_program is not None:
            print_optimized_prompts(optimized_program)

        # 8. Print summary
        print("\n" + "=" * 80)
        print("Experiment Summary")
        print("=" * 80)
        if base_score is not None:
            print(f"Base program score: {base_score:.2%}")
        if optimized_score is not None:
            print(f"Optimized program score: {optimized_score:.2%}")
            if base_score is not None:
                improvement = optimized_score - base_score
                print(f"Performance improvement: {improvement:+.2%}")

        print("\n✓ Experiment completed!")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()