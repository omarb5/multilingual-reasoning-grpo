# ==============================================================================
# config.py — Central configuration for all pipeline stages
# Edit this file to change models, hyperparameters, and paths.
# ==============================================================================

# ── Data Preparation ──────────────────────────────────────────────────────────
TOTAL_SAMPLES    = 1000
TEST_SPLIT       = 0.1

# Best open multilingual model on HF Inference Providers (Feb 2026)
# Alternatives: "Qwen/Qwen3-32B-Instruct", "deepseek-ai/DeepSeek-V3"
TRANSLATION_MODEL = "Qwen/Qwen3-235B-A22B-Instruct"
PROVIDER          = "auto"   # or: "together", "sambanova", "fireworks", "cerebras"

# ── Training ──────────────────────────────────────────────────────────────────
# Unsloth model — full bf16 on H100, no quantization
# Alternatives: "unsloth/Qwen3-14B-Instruct" (budget), "unsloth/Qwen3.5-35B-A3B" (MoE)
MODEL_NAME       = "unsloth/Qwen3.5-27B-Instruct"
LOAD_IN_4BIT     = False       # False = full bfloat16 (H100 80GB)
MAX_SEQ_LENGTH   = 2048
LORA_RANK        = 64

# GRPO hyperparameters (H100-scaled)
MAX_STEPS        = 500
NUM_GENERATIONS  = 8           # Parallel generations per prompt
BATCH_SIZE       = 1
GRAD_ACCUM       = 4
LEARNING_RATE    = 5e-6
WARMUP_RATIO     = 0.05
MAX_GRAD_NORM    = 0.1         # Critical for GRPO stability
GPU_MEMORY_UTIL  = 0.8

# ── Evaluation ────────────────────────────────────────────────────────────────
# Must be a model confirmed on HF Inference Providers
# (Qwen3-32B is widely available; swap to Qwen3.5-27B once available on providers)
BASELINE_MODEL   = "Qwen/Qwen3-32B-Instruct"
NUM_TEST_SAMPLES = 50
