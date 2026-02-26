# ==============================================================================
# config.py — Central configuration for all pipeline stages.
# Edit ONLY this file to change models, hyperparameters, and paths.
# ==============================================================================

# ── Data Preparation ──────────────────────────────────────────────────────────
TOTAL_SAMPLES    = 1000       # Total problems to collect (GSM8K + MATH)
TEST_SPLIT       = 0.1        # Fraction held out as test set

# HF Inference Providers — translation model
# Qwen3-235B-A22B is confirmed available on HF Inference Providers (Feb 2026)
TRANSLATION_MODEL = "Qwen/Qwen3-235B-A22B-Instruct"
PROVIDER          = "auto"    # or: "together", "sambanova", "fireworks", "cerebras"

# ── Training Model ────────────────────────────────────────────────────────────
# Qwen3-30B-A3B-Instruct-2507: 30B total params, 3B active (MoE)
# Requires ~64GB VRAM in bfloat16. Use H100 80GB.
# Source: Unsloth Qwen3_MoE notebook (unsloth/Qwen3-30B-A3B-Instruct-2507)
MODEL_NAME     = "unsloth/Qwen3-30B-A3B-Instruct-2507"
LOAD_IN_4BIT   = False        # 4-bit QLoRA NOT recommended for MoE (BnB unsupported)
MAX_SEQ_LENGTH = 2048
LORA_RANK      = 32           # Recommended value per notebook

# ── Custom GRPO Formatting Tags ───────────────────────────────────────────────
# These replace <think>/<answer> and are injected via the chat template.
# The SFT warmup teaches the model these tags before GRPO starts.
REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"

# ── SFT Warmup Hyperparameters (Step 1 of training) ──────────────────────────
# 50 steps on OpenMathReasoning-mini to teach format tags before GRPO.
# Values taken directly from Unsloth Qwen3_MoE notebook cell 27.
SFT_MAX_STEPS    = 50
SFT_LR           = 2e-4
SFT_WARMUP_STEPS = 5

# ── GRPO Hyperparameters (Step 2 of training) ─────────────────────────────────
GRPO_MAX_STEPS    = 500
NUM_GENERATIONS   = 4         # 4 is safe without vLLM on a 30B MoE model
BATCH_SIZE        = 1
GRAD_ACCUM        = 4         # Effective batch = BATCH_SIZE * GRAD_ACCUM = 4
LEARNING_RATE     = 5e-6
GRPO_WARMUP_STEPS = 5
MAX_GRAD_NORM     = 0.1       # Critical for GRPO stability

# ── Evaluation ────────────────────────────────────────────────────────────────
# IMPORTANT: Must be a model available on HF Inference Providers API.
# The local unsloth/Qwen3-30B-A3B-Instruct-2507 is NOT on Providers.
# Qwen3-235B-A22B-Instruct IS confirmed available.
BASELINE_MODEL   = "Qwen/Qwen3-235B-A22B-Instruct"
NUM_TEST_SAMPLES = 50
