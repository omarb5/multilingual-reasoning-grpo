import argparse
import os
import re
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOTrainer, GRPOConfig
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME")

# H100 Hardware Presets
MAX_SEQ_LENGTH = 2048        # Increased for H100
LORA_RANK = 64               # Higher rank for H100
LOAD_IN_4BIT = False         # Full bfloat16 on H100 for best quality

# GRPO Parameters (H100 Scaled)
MAX_STEPS = 500
NUM_GENERATIONS = 8          # H100 can handle 8 parallel generations easily
BATCH_SIZE = 1
GRAD_ACCUM = 4               
LEARNING_RATE = 5e-6

# Unsloth Model Catalog
MODEL_NAME = "unsloth/Qwen3.5-27B-Instruct"    

def get_reward_functions(target_lang, answer_lookup):
    def extract_xml_answer(text: str) -> str:
        ans = text.split("<answer>")[-1] if "<answer>" in text else ""
        return ans.split("</answer>")[0].strip() if "</answer>" in ans else ans.strip()

    def correctness_reward_func(prompts, completions, **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            pred = extract_xml_answer(text)
            p_text = prompt[-1]["content"] if isinstance(prompt, list) else prompt
            correct = answer_lookup.get(p_text, "")
            try:
                is_correct = abs(float(pred) - float(correct)) < 1e-6
            except:
                is_correct = pred.strip().upper() == correct.strip().upper()
            rewards.append(2.0 if is_correct else 0.0)
        return rewards

    def format_reward_func(completions, **kwargs) -> list[float]:
        pattern = r"^<reasoning>.*?</reasoning>\s*<answer>.*?</answer>$"
        rewards = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else comp
            rewards.append(1.0 if re.match(pattern, text, re.DOTALL | re.IGNORECASE) else 0.0)
        return rewards

    def language_purity_reward_func(completions, **kwargs) -> list[float]:
        rewards = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else comp
            if target_lang == "russian":
                tgt_chars = len(re.findall(r'[а-яА-Я]', text))
            elif target_lang == "arabic":
                tgt_chars = len(re.findall(r'[\u0600-\u06FF]', text))
            else:
                tgt_chars = 1
            eng_chars = len(re.findall(r'[a-zA-Z]', text))
            total = tgt_chars + eng_chars
            rewards.append(0.5 * (tgt_chars / total) if total > 0 else 0.0)
        return rewards

    return [correctness_reward_func, format_reward_func, language_purity_reward_func]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Training on H100")
    parser.add_argument("--lang", type=str, default="russian", help="Target language")
    args = parser.parse_args()

    if not HF_TOKEN or not HF_USERNAME:
        raise ValueError("Please set HF_TOKEN and HF_USERNAME in .env")

    save_hub_name = f"{HF_USERNAME}/grpo-reasoning-{args.lang}"
    dataset_name = f"{HF_USERNAME}/multilingual-reasoning-{args.lang}"

    print(f"🦥 Loading {MODEL_NAME} for H100 (bf16, no quantization)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        fast_inference=True,         # vLLM enabled
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.8,  # Safe for H100 80GB
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth", 
        random_state=42,
    )

    print(f"📊 Loading dataset from Hub: {dataset_name}")
    raw_data = load_dataset(dataset_name, split="train")

    q_key = f"question_{args.lang}"
    answer_lookup = {}
    formatted_data = []

    SYSTEM_PROMPT = """Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"""

    for item in raw_data:
        q = item.get(q_key)
        ans = str(item.get("answer_number", ""))
        if q and ans:
            formatted_data.append({
                "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
                "answer": ans
            })
            answer_lookup[q] = ans

    dataset = Dataset.from_list(formatted_data)
    
    training_args = GRPOConfig(
        output_dir=f"outputs_{args.lang}",
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        optim="adamw_8bit", # 8-bit optim saves VRAM
        logging_steps=5,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=512,
        max_completion_length=MAX_SEQ_LENGTH - 512,
        max_steps=MAX_STEPS,
        save_steps=250,
        report_to="none",
        use_vllm=True, 
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=get_reward_functions(args.lang, answer_lookup),
        args=training_args,
        train_dataset=dataset,
    )

    print(f"🚀 Starting GRPO Training for {args.lang} on H100...")
    trainer.train()

    print("💾 Saving and pushing to Hugging Face Hub...")
    model.push_to_hub_merged(save_hub_name, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
    model.push_to_hub_gguf(f"{save_hub_name}-GGUF", tokenizer, quantization_method=["q4_k_m"], token=HF_TOKEN)
    print(f"🎉 Successfully uploaded full model to https://huggingface.co/{save_hub_name}")