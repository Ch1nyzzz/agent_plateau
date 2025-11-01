
#!/bin/bash

set -e

LOG_DIR="/home/yuhan/ReAct_learning/agent_plateau/gepa-artifact/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/exp_hoverBench_Baseline_qwen3-8b_${TIMESTAMP}.log"
rm -rf /home/yuhan/ReAct_learning/agent_plateau/gepa-artifact/experiment_runs_data/experiment_runs/seed_0/hoverBench_HoverMultiHop_Baseline_qwen3-8b
(
  cd /home/yuhan/ReAct_learning/agent_plateau/gepa-artifact
  JAX_PLATFORMS=cpu python -m scripts.run_experiments \
    --bm_idx 0 \
    --benchmark_name "hoverBench" \
    --num_threads 32 \
    --program_idx 0 \
    --prog_name "HoverMultiHop" \
    --opt_idx 0 \
    --optim_name "Baseline" \
    --lm_config '{
      "name": "qwen3-8b",
      "model": "openai/arbor:qwen/qwen3-8b",
      "api_key": "API_KEY",
      "api_base": "http://localhost:{portnum}/v1/",
      "temperature": 0.6,
      "top_p": 0.95,
      "top_k": 20,
      "launch_kwargs": {
        "max_context_length": 8192
      },
      "train_kwargs": {
        "update_interval": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 20,
        "temperature": 0.6,
        "beta": 0.01,
        "learning_rate": 1e-05,
        "gradient_checkpointing": true,
        "gradient_checkpointing_kwargs": {
          "use_reentrant": false
        },
        "bf16": true,
        "lr_scheduler_type": "constant_with_warmup",
        "max_prompt_length": null,
        "max_completion_length": null,
        "scale_rewards": true,
        "max_grad_norm": 0.1,
        "lora": true,
        "report_to": "wandb",
        "log_completions": true,
        "logging_steps": 100,
        "generation_batch_size": 12
      }
    }' \
    --seed 0
) 2>&1 | tee "$LOG_FILE"
