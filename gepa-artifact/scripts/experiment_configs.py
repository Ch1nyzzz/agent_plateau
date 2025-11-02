import os
# 强制 JAX 使用 CPU，避免 CuDNN 版本不匹配问题
os.environ['JAX_PLATFORMS'] = 'cpu'

from pathlib import Path

BASE_EXPERIMENT_DIR = str((Path(__file__).parent.parent / "experiment_runs_data").resolve())

MAX_CONTEXT_LENGTH = 8192
MAX_CONTEXT_LENGTH_TRAINING = 8192
LAUNCH_KWARGS = {
    "max_context_length": MAX_CONTEXT_LENGTH
}
SAMPLING_TEMPERATURE = 0.6
TRAIN_KWARGS_GRPO_DEFAULT = {
    "update_interval": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 20,
    "temperature": SAMPLING_TEMPERATURE,
    "beta": 0.01,
    "learning_rate": 1e-5,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "bf16": True,
    "lr_scheduler_type": "constant_with_warmup",
    "max_prompt_length": None,
    "max_completion_length": None,
    "scale_rewards": True,
    "max_grad_norm": 0.1,
    "lora": True,
    'report_to': "wandb",
    'log_completions': True,
    'logging_steps': 100,
    "generation_batch_size": 12,
}
TRAIN_KWARGS_GRPO_QWEN = {**TRAIN_KWARGS_GRPO_DEFAULT}

# Add/modify available LMs here.
LM_CONFIGS = [
    # {
    #     "name": "qwen3-8b",
    #     "model": "openai/arbor:qwen/qwen3-8b",
    #     "api_key": "API_KEY",
    #     "api_base": "http://localhost:{portnum}/v1/",
    #     "temperature": 0.6,
    #     "top_p": 0.95,
    #     "top_k": 20,
    #     "launch_kwargs": LAUNCH_KWARGS,
    #     "train_kwargs": TRAIN_KWARGS_GRPO_QWEN,
    # },
    {
        "name": "gpt-41-mini",
        "model": "openai/gpt-4.1-mini-2025-04-14",
        "api_key": "env:OPENAI_API_KEY",
        "temperature": 1.0,
    },
]

def get_benchmarks():
    from gepa_artifact.benchmarks.hover import benchmark as hover_metas
    from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpotQA_metas
    from gepa_artifact.benchmarks.papillon import benchmark as papillon_metas
    from gepa_artifact.benchmarks.IFBench import benchmark as ifbench_metas
    from gepa_artifact.benchmarks.livebench_math import benchmark as math_metas
    # from gepa_artifact.benchmarks.AIME import benchmark as aime_metas

    from gepa_artifact.utils.optimizers import OptimizerConfig

    benchmark_metas = hover_metas + hotpotQA_metas + papillon_metas + ifbench_metas + math_metas # + aime_metas
    return benchmark_metas

def get_optimizers():
    import dspy
    from gepa_artifact.gepa.gepa import GEPA
    from dspy.teleprompt.grpo import GRPO
    from gepa_artifact.utils.optimizers import OptimizerConfig
    optimizers = [
        ("Baseline",
            OptimizerConfig(
                optimizer=None,
                init_args={},
                compile_args={},
                langProBe_configs=dict(
                    launch_arbor=True,
                ),
                name="Baseline",
            )
        ),
        (
            "GEPA-5",
            OptimizerConfig(
                optimizer=GEPA,
                init_args=dict(
                    run_linearized_gepa=False,
                    use_merge=False,
                    set_for_merge_minibatch='val',
                    track_scores_on='val',
                    num_iters=5,
                ),
                compile_args=dict(),
                langProBe_configs=dict(
                    use_valset=True,
                    launch_arbor=True,
                    use_cache_from_opt="Baseline",
                ),
                name="GEPA-5",
            )
        ),
        (
            "GEPA-10",
            OptimizerConfig(
                optimizer=GEPA,
                init_args=dict(
                    run_linearized_gepa=False,
                    use_merge=False,
                    set_for_merge_minibatch='val',
                    track_scores_on='val',
                    num_iters=10,
                ),
                compile_args=dict(),
                langProBe_configs=dict(
                    use_valset=True,
                    launch_arbor=True,
                    use_cache_from_opt="Baseline",
                ),
                name="GEPA-10",
            )
        ),
        (
            "GEPA-15",
            OptimizerConfig(
                optimizer=GEPA,
                init_args=dict(
                    run_linearized_gepa=False,
                    use_merge=False,
                    set_for_merge_minibatch='val',
                    track_scores_on='val',
                    num_iters=15,
                ),
                compile_args=dict(),
                langProBe_configs=dict(
                    use_valset=True,
                    launch_arbor=True,
                    use_cache_from_opt="Baseline",
                ),
                name="GEPA-15",
            )
        ),
        (
            "GEPA-20",
            OptimizerConfig(
                optimizer=GEPA,
                init_args=dict(
                    run_linearized_gepa=False,
                    use_merge=False,
                    set_for_merge_minibatch='val',
                    track_scores_on='val',
                    num_iters=20,
                ),
                compile_args=dict(),
                langProBe_configs=dict(
                    use_valset=True,
                    launch_arbor=True,
                    use_cache_from_opt="Baseline",
                ),
                name="GEPA-20",
            )
        ),
        (
            "GEPA-25",
            OptimizerConfig(
                optimizer=GEPA,
                init_args=dict(
                    run_linearized_gepa=False,
                    use_merge=False,
                    set_for_merge_minibatch='val',
                    track_scores_on='val',
                    num_iters=25,
                ),
                compile_args=dict(),
                langProBe_configs=dict(
                    use_valset=True,
                    launch_arbor=True,
                    use_cache_from_opt="Baseline",
                ),
                name="GEPA-25",
            )
        ),
        (
            "GEPA-50",
            OptimizerConfig(
                optimizer=GEPA,
                init_args=dict(
                    run_linearized_gepa=False,
                    use_merge=False,
                    set_for_merge_minibatch='val',
                    track_scores_on='val',
                    num_iters=50,
                ),
                compile_args=dict(),
                langProBe_configs=dict(
                    use_valset=True,
                    launch_arbor=True,
                    use_cache_from_opt="Baseline",
                ),
                name="GEPA-50",
            )
        ),
    ]

    return optimizers

def get_max_invocations(bench, prog, model, opt):
    known_max_calls = {
        ('HotpotQABench', 'HotpotMultiHop', 'MIPROv2-Heavy'): 6871,
        ('Papillon', 'PAPILLON', 'MIPROv2-Heavy'): 2426,
        ('hoverBench', 'HoverMultiHop', 'MIPROv2-Heavy'): 7051,
        ('IFBench', 'IFBenchCoT2StageProgram', 'MIPROv2-Heavy'): 3593,
        ('LiveBenchMathBench', 'CoT', 'MIPROv2-Heavy'): 1839,
        ('AIMEBench', 'CoT', 'MIPROv2-Heavy'): 1839,
    }

    if (bench, prog, opt) in known_max_calls:
        return known_max_calls[(bench, prog, opt)]

    raise Exception(
        f"Could not find max invocations for {bench}, {prog}, {opt}. "
        "Please add it to the known_max_calls dictionary in get_max_invocations."
    )
