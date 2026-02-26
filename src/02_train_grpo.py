# ==============================================================================
# 02_train_grpo.py — Full training pipeline: SFT warmup → GRPO
#
# Based on the Unsloth Qwen3_MoE notebook:
#   https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_MoE.ipynb
#
# Pipeline:
#   Step 1 — SFT warmup  : 50 steps on OpenMathReasoning-mini
#                           Teaches the model the custom format tags so GRPO
#                           doesn't waste reward signal on formatting.
#   Step 2 — GRPO        : Trains multilingual reasoning via 3 reward functions.
#
# Run: python src/02_train_grpo.py --lang russian
# ==============================================================================

import argparse
import gc
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BATCH_SIZE, GRAD_ACCUM, GRPO_MAX_STEPS, GRPO_WARMUP_STEPS,
    LEARNING_RATE, LOAD_IN_4BIT, LORA_RANK, MAX_GRAD_NORM,
    MAX_SEQ_LENGTH, MODEL_NAME, NUM_GENERATIONS,
    REASONING_END, REASONING_START, SFT_LR, SFT_MAX_STEPS, SFT_WARMUP_STEPS,
    SOLUTION_END, SOLUTION_START,
)

load_dotenv()
HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]

# System prompt — matches notebook cell 11 verbatim.
# The final line intentionally shows '<SOLUTION></SOLUTION>' as a visual placeholder;
# the model is expected to fill in its answer inside those tags.
SYSTEM_PROMPT = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
)


def build_chat_template(system_prompt: str, reasoning_start: str) -> str:
    """
    Builds the Jinja2 chat template using .replace() injection — exactly as in
    Unsloth Qwen3_MoE notebook cell 13. This is the ONLY correct way to embed
    Python variable values into a Jinja2 template string without breaking the
    template syntax. Do NOT use f-strings or triple-quote interpolation here.
    """
    # Build with literal placeholder strings first
    template = (
        "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
        "{% else %}"
            "{{ '{system_prompt}' + eos_token }}"
            "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
                "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] + eos_token }}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
        "{% endif %}"
    )
    # Inject actual values via .replace() — notebook cell 13 pattern
    template = template \
        .replace("'{system_prompt}'",   f"'{system_prompt}'") \
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    return template


def extract_answer(text: str) -> str:
    """Extract the answer from <SOLUTION>...</SOLUTION> tags."""
    if SOLUTION_START not in text:
        return ""
    ans = text.split(SOLUTION_START)[-1]
    return ans.split(SOLUTION_END)[0].strip() if SOLUTION_END in ans else ans.strip()


def get_reward_functions(target_lang: str, answer_lookup: dict) -> list:
    """Returns the three GRPO reward functions."""

    def correctness_reward(prompts, completions, **kwargs) -> list[float]:
        """2.0 pts if the extracted answer matches ground truth."""
        rewards = []
        for prompt, completion in zip(prompts, completions):
            text    = completion[0]["content"] if isinstance(completion, list) else str(completion)
            pred    = extract_answer(text)
            p_text  = str(prompt[-1]["content"]) if isinstance(prompt, list) else str(prompt)
            correct = answer_lookup.get(p_text, "")
            try:
                ok = abs(float(pred) - float(correct)) < 1e-6
            except (ValueError, TypeError):
                ok = pred.strip().upper() == correct.strip().upper()
            rewards.append(2.0 if ok else 0.0)
        return rewards

    def format_reward(completions, **kwargs) -> list[float]:
        """1.0 pt if response follows <start_working_out>...<SOLUTION>...</SOLUTION>."""
        rs  = re.escape(REASONING_START)
        re_ = re.escape(REASONING_END)
        ss  = re.escape(SOLUTION_START)
        se  = re.escape(SOLUTION_END)
        pattern = rf"^{rs}.*?{re_}\s*{ss}.*?{se}$"
        rewards = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else str(comp)
            rewards.append(1.0 if re.match(pattern, text, re.DOTALL) else 0.0)
        return rewards

    def language_purity_reward(completions, **kwargs) -> list[float]:
        """Up to 0.5 pts for responding in the target language."""
        rewards = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else str(comp)
            if target_lang == "russian":
                tgt = len(re.findall(r'[\u0400-\u04FF]', text))
            elif target_lang == "arabic":
                tgt = len(re.findall(r'[\u0600-\u06FF]', text))
            else:
                tgt = max(1, len(text) // 2)
            eng   = len(re.findall(r'[a-zA-Z]', text))
            total = tgt + eng
            rewards.append(0.5 * (tgt / total) if total > 0 else 0.0)
        return rewards

    return [correctness_reward, format_reward, language_purity_reward]


# ==============================================================================
# STEP 1: SFT WARMUP  (Qwen3_MoE notebook cells 16–32)
# ==============================================================================

def run_sft_warmup(model, tokenizer) -> None:
    """
    Pre-fine-tunes the model on OpenMathReasoning-mini for SFT_MAX_STEPS steps.
    This teaches the custom <start_working_out>/<SOLUTION> format so GRPO can
    focus purely on reasoning quality rather than learning to produce the tags.

    Directly replicates notebook cells 17-32.
    """
    print("\n\U0001f4da Step 1: SFT warmup — teaching custom format tags...")

    # Load dataset — notebook cell 17
    raw = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    df  = raw.to_pandas()[["expected_answer", "problem", "generated_solution"]]

    # Keep only numeric answers — notebook cell 17
    is_number = pd.to_numeric(pd.Series(df["expected_answer"]), errors="coerce").notnull()
    df = df.iloc[np.where(is_number)[0]].copy()

    # Format each row — notebook cell 19
    def format_row(row):
        thoughts = row["generated_solution"] \
            .replace("<think>", "") \
            .replace("</think>", "") \
            .strip()
        assistant_content = (
            REASONING_START + thoughts + REASONING_END +
            SOLUTION_START  + str(row["expected_answer"]) + SOLUTION_END
        )
        return [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": str(row["problem"])},
            {"role": "assistant", "content": assistant_content},
        ]

    df["Messages"] = df.apply(format_row, axis=1)

    # Truncate to max_seq_length / 2 — notebook cell 23
    def count_tokens(msgs):
        encoded = tokenizer.apply_chat_template(msgs)
        return len(encoded) if isinstance(encoded, list) else len(encoded["input_ids"])

    df["N"] = df["Messages"].apply(count_tokens)
    df = df.loc[df["N"] <= MAX_SEQ_LENGTH // 2].copy()
    print(f"   SFT dataset: {len(df)} examples (filtered to \u2264{MAX_SEQ_LENGTH // 2} tokens each)")

    # Tokenize — notebook cell 25
    df["text"] = tokenizer.apply_chat_template(
        df["Messages"].values.tolist(), tokenize=False
    )
    sft_dataset = Dataset.from_pandas(df)

    # Train — notebook cell 27
    sft_trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = sft_dataset,
        args = SFTConfig(
            dataset_text_field          = "text",
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1,
            warmup_steps                = SFT_WARMUP_STEPS,
            max_steps                   = SFT_MAX_STEPS,
            learning_rate               = SFT_LR,
            logging_steps               = 5,
            optim                       = "adamw_8bit",  # matches notebook exactly
            weight_decay                = 0.001,
            lr_scheduler_type           = "linear",
            seed                        = 3407,           # matches notebook exactly
            report_to                   = "none",
        ),
    )
    sft_trainer.train()
    print("   \u2705 SFT warmup complete.")

    # Free memory before GRPO — notebook cell 32
    del sft_dataset
    torch.cuda.empty_cache()
    gc.collect()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Training for Multilingual Reasoning")
    parser.add_argument("--lang",      type=str,  default="russian",      help="Target language")
    parser.add_argument("--model",     type=str,  default=MODEL_NAME,     help="Unsloth model name")
    parser.add_argument("--steps",     type=int,  default=GRPO_MAX_STEPS, help="GRPO training steps")
    parser.add_argument("--skip_sft",  action="store_true",               help="Skip SFT warmup (not recommended)")
    args = parser.parse_args()

    if not HF_TOKEN or not HF_USERNAME:
        raise EnvironmentError("HF_TOKEN and HF_USERNAME must be set in .env")

    save_hub_name = f"{HF_USERNAME}/grpo-reasoning-{args.lang}"
    dataset_name  = f"{HF_USERNAME}/multilingual-reasoning-{args.lang}"
    q_key         = f"question_{args.lang}"

    # ── 1. Load model — notebook cell 9
    print(f"\n\U0001f9a5 Loading {args.model}...")
    print("   Requires ~64 GB VRAM (model is ~60 GB in bfloat16).")
    # IMPORTANT: Only model_name, max_seq_length, load_in_4bit, fast_inference are valid
    # kwargs in non-vLLM mode. Do NOT pass max_lora_rank or gpu_memory_utilization.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.model,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit   = LOAD_IN_4BIT,
        fast_inference = False,  # MoE fast_inference not yet supported (per notebook)
    )

    # ── 2. Set chat template — notebook cell 13 (.replace() pattern)
    tokenizer.chat_template = build_chat_template(SYSTEM_PROMPT, REASONING_START)

    # ── 3. Apply LoRA — notebook cell 9
    # gate_up_proj is included to enable LoRA on MoE routing layers.
    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_RANK,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "gate_up_proj",  # MoE layers
        ],
        lora_alpha               = LORA_RANK * 2,  # *2 speeds up training (per notebook)
        use_gradient_checkpointing = True,         # Reduces VRAM usage (per notebook)
        random_state             = 3407,           # Matches notebook exactly
        bias                     = "none",
    )

    # ── 4. SFT warmup — notebook cells 16-32
    if not args.skip_sft:
        run_sft_warmup(model, tokenizer)
    else:
        print("\n\u26a0\ufe0f  Skipping SFT warmup. GRPO will need more steps to learn format.")

    # ── 5. Load multilingual GRPO dataset from HF Hub
    print(f"\n\U0001f4ca Loading GRPO dataset: {dataset_name}")
    raw_data       = load_dataset(dataset_name, split="train", token=HF_TOKEN)
    answer_lookup  = {}
    formatted_data = []

    for item in raw_data:
        q   = str(item.get(q_key, "")).strip()
        ans = str(item.get("answer_number", "")).strip()
        if q and ans and ans.lower() != "none":
            formatted_data.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": q},
                ],
                "answer": ans,
            })
            answer_lookup[q] = ans

    if not formatted_data:
        raise ValueError(
            f"No valid data found in {dataset_name}.\n"
            "Make sure 01_prepare_data.py ran successfully and pushed the dataset."
        )

    grpo_dataset = Dataset.from_list(formatted_data)
    print(f"   \u2705 {len(grpo_dataset)} prompts ready for GRPO.")

    # ── 6. GRPO training
    training_args = GRPOConfig(
        output_dir                   = f"outputs_{args.lang}",
        learning_rate                = LEARNING_RATE,
        lr_scheduler_type            = "cosine",
        # adamw_torch_fused is best on H100 (fused CUDA kernel, lower overhead).
        # Fallback: "adamw_8bit" if you see CUDA errors.
        optim                        = "adamw_torch_fused",
        warmup_steps                 = GRPO_WARMUP_STEPS,
        max_grad_norm                = MAX_GRAD_NORM,
        logging_steps                = 5,
        per_device_train_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACCUM,
        num_generations              = NUM_GENERATIONS,
        max_prompt_length            = 512,
        max_completion_length        = MAX_SEQ_LENGTH - 512,
        max_steps                    = args.steps,
        save_steps                   = 250,
        report_to                    = "none",
        use_vllm                     = False,  # MoE fast_inference not yet supported
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = get_reward_functions(args.lang, answer_lookup),
        args             = training_args,
        train_dataset    = grpo_dataset,
    )

    print(f"\n\U0001f680 Step 2: GRPO training [{args.lang}] for {args.steps} steps...")
    trainer.train()
    print("   \u2705 GRPO training complete.")

    # ── 7. Push merged model + GGUF to HF Hub
    print("\n\U0001f4be Merging LoRA weights and uploading to HF Hub...")
    model.push_to_hub_merged(
        save_hub_name, tokenizer,
        save_method = "merged_16bit",
        token       = HF_TOKEN,
    )
    model.push_to_hub_gguf(
        f"{save_hub_name}-GGUF", tokenizer,
        quantization_method = ["q4_k_m"],
        token               = HF_TOKEN,
    )
    print(f"\n\U0001f389 Done! Model live at https://huggingface.co/{save_hub_name}")
