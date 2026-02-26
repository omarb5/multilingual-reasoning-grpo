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
    REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END
)

load_dotenv()
HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]

# NEW: Structured reasoning system prompt based on Unsloth Qwen3_MoE notebook
SYSTEM_PROMPT = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {REASONING_START} and {REASONING_END}.
Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"""


def extract_custom_answer(text: str) -> str:
    """Extracts answer using the new <SOLUTION> formatting"""
    if SOLUTION_START not in text:
        return ""
    ans = text.split(SOLUTION_START)[-1]
    return ans.split(SOLUTION_END)[0].strip() if SOLUTION_END in ans else ans.strip()


def get_reward_functions(target_lang: str, answer_lookup: dict):
    def correctness_reward(prompts, completions, **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            text    = completion[0]["content"] if isinstance(completion, list) else completion
            pred    = extract_custom_answer(text)
            p_text  = str(prompt[-1]["content"]) if isinstance(prompt, list) else str(prompt)
            correct = answer_lookup.get(p_text, "")
            try:
                ok = abs(float(pred) - float(correct)) < 1e-6
            except (ValueError, TypeError):
                ok = pred.strip().upper() == correct.strip().upper()
            rewards.append(2.0 if ok else 0.0)
        return rewards

    def format_reward(completions, **kwargs) -> list[float]:
        # Expecting: <start_working_out>...<end_working_out><SOLUTION>...</SOLUTION>
        pattern = rf"^{REASONING_START}.*?{REASONING_END}\s*{SOLUTION_START}.*?{SOLUTION_END}$"
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

    # ── 1. Load model 
    print(f"🦥 Loading {args.model} in bfloat16 (H100)...")
    # Note: Qwen3-30B-A3B is very large. fast_inference=False is set in the 
    # official notebook because it is "Not supported for MoE (yet!)"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name           = args.model,
        max_seq_length       = MAX_SEQ_LENGTH,
        load_in_4bit         = LOAD_IN_4BIT,
        dtype                = torch.bfloat16,
        fast_inference       = False,               # NEW: MoE fast inference not supported yet
        max_lora_rank        = LORA_RANK,
        gpu_memory_utilization = GPU_MEMORY_UTIL,
    )

    # Custom Chat Template mapping to our Reasoning/Solution tags
    chat_template = """{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + eos_token }}{% set loop_messages = messages[1:] %}{% else %}{{ '""" + SYSTEM_PROMPT + """' + eos_token }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '""" + REASONING_START + """' }}{% endif %}"""
    tokenizer.chat_template = chat_template

    model = FastLanguageModel.get_peft_model(
        model,
        r                        = LORA_RANK,
        # NEW: target_modules updated to include MoE specific layers per Unsloth guide
        target_modules           = ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj", "gate_up_proj"],
        lora_alpha               = LORA_RANK * 2,   # *2 speeds up training per notebook
        lora_dropout             = 0,
        use_gradient_checkpointing = True,          # Changed from "unsloth" to True per notebook
        random_state             = 42,
        bias                     = "none",
    )

    torch.cuda.empty_cache()

    # ── 2. Load dataset from HF Hub
    print(f"📊 Loading dataset: {dataset_name}")
    raw_data       = load_dataset(dataset_name, split="train")
    answer_lookup  = {}
    formatted_data = []

    for item in raw_data:
        q   = str(item.get(q_key, ""))
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
        optim                        = "adamw_torch_fused",
        warmup_steps                 = 5,                     # Replaced warmup_ratio per notebook
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
        use_vllm                     = False,                 # Disabled because fast_inference is False for MoE
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