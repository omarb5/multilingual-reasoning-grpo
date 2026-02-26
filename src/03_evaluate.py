# ==============================================================================
# 03_evaluate.py — Evaluate trained model vs baseline via HF Inference API
# Run: python src/03_evaluate.py --lang russian
# ==============================================================================

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from huggingface_hub import InferenceClient
from unsloth import FastLanguageModel
from datasets import load_dataset
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASELINE_MODEL, NUM_TEST_SAMPLES, PROVIDER, REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END

load_dotenv()
HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]

# NEW: Structured reasoning system prompt
SYSTEM_PROMPT = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {REASONING_START} and {REASONING_END}.
Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"""


def extract_answer(text: str) -> str:
    """Extracts answer using the new <SOLUTION> formatting"""
    if SOLUTION_START in text:
        ans = text.split(SOLUTION_START)[-1]
        return ans.split(SOLUTION_END)[0].strip() if SOLUTION_END in ans else ans.strip()
    
    # Fallbacks if format fails
    for pat in [r"####\s*([+-]?\d+\.?\d*)", r"\\boxed\{([^}]+)\}"]:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    nums = re.findall(r"[+-]?\d+\.?\d*", text)
    return nums[-1] if nums else "None"


def is_correct(pred: str, true_ans: str) -> bool:
    if pred == "None":
        return False
    try:
        return abs(float(pred) - float(true_ans)) < 1e-6
    except (ValueError, TypeError):
        return pred.strip().upper() == str(true_ans).strip().upper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multilingual Reasoning Model")
    parser.add_argument("--lang",     type=str, default="russian")
    parser.add_argument("--baseline", type=str, default=BASELINE_MODEL, help="HF model ID for baseline comparison")
    parser.add_argument("--samples",  type=int, default=NUM_TEST_SAMPLES)
    args = parser.parse_args()

    if not HF_TOKEN or not HF_USERNAME:
        raise EnvironmentError("HF_TOKEN and HF_USERNAME must be set in .env")

    dataset_name       = f"{HF_USERNAME}/multilingual-reasoning-{args.lang}"
    trained_model_name = f"{HF_USERNAME}/grpo-reasoning-{args.lang}"
    q_key              = f"question_{args.lang}"

    # ── 1. Load test data from HF Hub
    print(f"📊 Loading test split from {dataset_name}...")
    test_data     = load_dataset(dataset_name, split="test")
    test_problems = [item for item in test_data if item.get(q_key)][:args.samples]
    print(f"   {len(test_problems)} problems loaded.")

    # ── 2. Evaluate GRPO-trained model (loaded locally from Hub)
    print(f"\n🦥 Evaluating {trained_model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = trained_model_name,
        max_seq_length = 2048,
        load_in_4bit   = True,   # 4-bit is fine for inference
    )
    FastLanguageModel.for_inference(model)

    # Custom Chat Template mapping to our Reasoning/Solution tags
    chat_template = """{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + eos_token }}{% set loop_messages = messages[1:] %}{% else %}{{ '""" + SYSTEM_PROMPT + """' + eos_token }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '""" + REASONING_START + """' }}{% endif %}"""
    tokenizer.chat_template = chat_template

    grpo_results = []

    for item in tqdm(test_problems, desc="GRPO"):
        prompt   = str(item[q_key])
        true_ans = str(item["answer_number"])
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        text     = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs   = tokenizer([text], return_tensors="pt").to("cuda")
        outputs  = model.generate(
            **inputs,
            max_new_tokens = 1024,
            do_sample      = True,                          
            temperature    = 0.6,
            pad_token_id   = tokenizer.eos_token_id,        
        )
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text):].strip()
        pred = extract_answer(resp)
        grpo_results.append({
            "id": item["id"], "source": item["source"],
            "response": resp, "predicted": pred, "actual": true_ans,
            "correct": is_correct(pred, true_ans),
        })

    # ── 3. Evaluate baseline via HF Inference Providers 
    print(f"\n🌐 Evaluating baseline [{args.baseline}] via HF Inference API...")
    client       = InferenceClient(api_key=HF_TOKEN)
    base_results = []

    for item in tqdm(test_problems, desc="Baseline"):
        prompt   = str(item[q_key])
        true_ans = str(item["answer_number"])
        try:
            resp = client.chat.completions.create(
                model    = args.baseline,
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                max_tokens  = 1024,
                temperature = 0.6,
                provider    = PROVIDER,
            )
            resp_text = resp.choices[0].message.content
        except Exception as e:
            print(f"  API Error for {item['id']}: {e}")
            resp_text = ""
        pred = extract_answer(resp_text)
        base_results.append({
            "id": item["id"], "source": item["source"],
            "response": resp_text, "predicted": pred, "actual": true_ans,
            "correct": is_correct(pred, true_ans),
        })

    # ── 4. Metrics & plot
    grpo_acc = sum(r["correct"] for r in grpo_results) / len(grpo_results)
    base_acc = sum(r["correct"] for r in base_results) / len(base_results)

    print(f"\n{'='*40}")
    print(f"  GRPO Trained : {grpo_acc:.1%}")
    print(f"  Baseline     : {base_acc:.1%}")
    print(f"  Delta        : {(grpo_acc - base_acc):+.1%}")
    print(f"{'='*40}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([f"Baseline\n({args.baseline.split('/')[-1]})", "GRPO Trained"],
           [base_acc, grpo_acc], color=["#94a3b8", "#38bdf8"])
    ax.set_title(f"Qwen3 MoE Reasoning — {args.lang.title()}")
    ax.set_ylim([0, 1.0])
    for i, v in enumerate([base_acc, grpo_acc]):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/eval_{args.lang}.png", dpi=150, bbox_inches="tight")
    pd.DataFrame(grpo_results).to_csv(f"results/grpo_results_{args.lang}.csv", index=False)
    pd.DataFrame(base_results).to_csv(f"results/baseline_results_{args.lang}.csv", index=False)
    print("✅ Saved results to results/")