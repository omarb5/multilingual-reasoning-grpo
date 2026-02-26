# ==============================================================================
# 02_train_grpo.py — GRPO training on H100 with Unsloth + TRL
# Run: python src/02_train_grpo.py --lang russian
# ==============================================================================

import argparse
import os
import re
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_NAME, LOAD_IN_4BIT, MAX_SEQ_LENGTH, LORA_RANK,
    MAX_STEPS, NUM_GENERATIONS, BATCH_SIZE, GRAD_ACCUM,
    LEARNING_RATE, WARMUP_RATIO, MAX_GRAD_NORM, GPU_MEMORY_UTIL,
)

load_dotenv()
HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]

# Structured reasoning system prompt (FIX 7: actual newlines, not \\n literals)
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""


def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text:
        return ""
    ans = text.split("<answer>")[-1]
    return ans.split("</answer>")[0].strip() if "</answer>" in ans else ans.strip()


def get_reward_functions(target_lang: str, answer_lookup: dict):
    def correctness_reward(prompts, completions, **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            text    = completion[0]["content"] if isinstance(completion, list) else completion
            pred    = extract_xml_answer(text)
            p_text  = str(prompt[-1]["content"]) if isinstance(prompt, list) else str(prompt)  # FIX 8
            correct = answer_lookup.get(p_text, "")
            try:
                ok = abs(float(pred) - float(correct)) < 1e-6
            except (ValueError, TypeError):
                ok = pred.strip().upper() == correct.strip().upper()
            rewards.append(2.0 if ok else 0.0)
        return rewards

    def format_reward(completions, **kwargs) -> list[float]:
        pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>$"
        rewards = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else comp
            rewards.append(1.0 if re.match(pattern, text, re.DOTALL | re.IGNORECASE) else 0.0)
        return rewards

    def language_purity_reward(completions, **kwargs) -> list[float]:
        rewards = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else comp
            if target_lang == "russian":
                tgt = len(re.findall(r'[\u0400-\u04FF]', text))
            elif target_lang == "arabic":
                tgt = len(re.findall(r'[\u0600-\u06FF]', text))
            else:
                tgt = 1
            eng   = len(re.findall(r'[a-zA-Z]', text))
            total = tgt + eng
            rewards.append(0.5 * (tgt / total) if total > 0 else 0.0)
        return rewards

    return [correctness_reward, format_reward, language_purity_reward]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument("--lang",  type=str, default="russian")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    if not HF_TOKEN or not HF_USERNAME:
        raise EnvironmentError("HF_TOKEN and HF_USERNAME must be set in .env")

    save_hub_name  = f"{HF_USERNAME}/grpo-reasoning-{args.lang}"
    dataset_name   = f"{HF_USERNAME}/multilingual-reasoning-{args.lang}"
    q_key          = f"question_{args.lang}"

    # ── 1. Load model (FIX 9: explicit dtype=bfloat16 for full-precision H100)
    print(f"🦥 Loading {args.model} in bfloat16 (H100)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name           = args.model,
        max_seq_length       = MAX_SEQ_LENGTH,
        load_in_4bit         = LOAD_IN_4BIT,
        dtype                = torch.bfloat16,     # FIX 9
        fast_inference       = True,
        max_lora_rank        = LORA_RANK,
        gpu_memory_utilization = GPU_MEMORY_UTIL,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                        = LORA_RANK,
        target_modules           = ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
        lora_alpha               = LORA_RANK,
        lora_dropout             = 0,
        use_gradient_checkpointing = "unsloth",
        random_state             = 42,
    )

    # FIX 10: clear cache before vLLM initializes
    torch.cuda.empty_cache()

    # ── 2. Load dataset from HF Hub
    print(f"📊 Loading dataset: {dataset_name}")
    raw_data       = load_dataset(dataset_name, split="train")
    answer_lookup  = {}
    formatted_data = []

    for item in raw_data:
        q   = str(item.get(q_key, ""))   # FIX 8: force str, avoid Arrow type issues
        ans = str(item.get("answer_number", ""))
        if q and ans:
            formatted_data.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": q},
                ],
                "answer": ans,
            })
            answer_lookup[q] = ans

    dataset = Dataset.from_list(formatted_data)
    print(f"✅ Dataset ready: {len(dataset)} prompts")

    # ── 3. Train
    training_args = GRPOConfig(
        output_dir                   = f"outputs_{args.lang}",
        learning_rate                = LEARNING_RATE,
        lr_scheduler_type            = "cosine",
        optim                        = "adamw_torch_fused",   # FIX 11: faster on H100
        warmup_ratio                 = WARMUP_RATIO,          # FIX 13
        max_grad_norm                = MAX_GRAD_NORM,         # FIX 12: 0.1 for GRPO stability
        logging_steps                = 5,
        per_device_train_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACCUM,
        num_generations              = NUM_GENERATIONS,
        max_prompt_length            = 512,
        max_completion_length        = MAX_SEQ_LENGTH - 512,
        max_steps                    = args.steps,
        save_steps                   = 250,
        report_to                    = "none",
        use_vllm                     = True,
    )

    trainer = GRPOTrainer(
        model             = model,
        processing_class  = tokenizer,
        reward_funcs      = get_reward_functions(args.lang, answer_lookup),
        args              = training_args,
        train_dataset     = dataset,
    )

    print(f"🚀 Starting GRPO training for [{args.lang}] on H100...")
    trainer.train()

    # ── 4. Push to Hub
    print("💾 Merging LoRA and pushing to Hugging Face Hub...")
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
    print(f"🎉 Model live at https://huggingface.co/{save_hub_name}")
