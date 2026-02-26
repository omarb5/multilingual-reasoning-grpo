# ==============================================================================
# config.py — Central configuration for all pipeline stages
# Edit this file to change models, hyperparameters, and paths.
# ==============================================================================

# ── Data Preparation ──────────────────────────────────────────────────────────
TOTAL_SAMPLES    = 1000
TEST_SPLIT       = 0.1

# Best open multilingual model on HF Inference Providers (Feb 2026)
TRANSLATION_MODEL = "Qwen/Qwen3-235B-A22B-Instruct"
PROVIDER          = "auto"   # or: "together", "sambanova", "fireworks", "cerebras"

# ── Training ──────────────────────────────────────────────────────────────────
# NEW: Based on Unsloth's Qwen3_MoE notebook, we use Qwen3-30B-A3B-Instruct-2507
MODEL_NAME       = "unsloth/Qwen3-30B-A3B-Instruct-2507"
LOAD_IN_4BIT     = False       # False = full bfloat16 (H100 80GB)
MAX_SEQ_LENGTH   = 2048
LORA_RANK        = 32          # 32 recommended for MoE in the official notebook

# GRPO Custom Formatting (as per Qwen3_MoE notebook)
REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"

# GRPO hyperparameters (H100-scaled)
MAX_STEPS        = 500
NUM_GENERATIONS  = 8           # Parallel generations per prompt
BATCH_SIZE       = 1
GRAD_ACCUM       = 4
LEARNING_RATE    = 5e-6
WARMUP_RATIO     = 0.05
MAX_GRAD_NORM    = 0.1         
GPU_MEMORY_UTIL  = 0.8

# ── Evaluation ────────────────────────────────────────────────────────────────
# Baseline model for testing
BASELINE_MODEL   = "Qwen/Qwen3-30B-A3B-Instruct-2507"
NUM_TEST_SAMPLES = 50