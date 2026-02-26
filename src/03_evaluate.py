# ==============================================================================
# 03_evaluate.py — Evaluate trained model vs baseline via HF Inference API
# Run: python src/03_evaluate.py --lang russian
# ==============================================================================

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from tqdm.auto import tqdm
from unsloth import FastLanguageModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASELINE_MODEL, NUM_TEST_SAMPLES, PROVIDER,
    REASONING_END, REASONING_START, SOLUTION_END, SOLUTION_START,
)

load_dotenv()
HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]

# Must exactly match the system prompt used during training in 02_train_grpo.py
SYSTEM_PROMPT = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
)


def build_chat_template(system_prompt: str, reasoning_start: str) -> str:
    """
    Identical to 02_train_grpo.py — the chat template MUST match training exactly,
    otherwise the model will not produce answers in the expected format.
    Uses .replace() injection as per Unsloth Qwen3_MoE notebook cell 13.
    """
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
    template = template \
        .replace("'{system_prompt}'",   f"'{system_prompt}'") \
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    return template


def extract_answer(text: str) -> str:
    """
    Extracts the answer from <SOLUTION>...</SOLUTION> tags.
    Falls back to #### and \\boxed{} patterns for baseline models that
    don't use our custom tags.
    """
    if SOLUTION_START in text:
        ans = text.split(SOLUTION_START)[-1]
        raw = ans.split(SOLUTION_END)[0].strip() if SOLUTION_END in ans else ans.strip()
        return raw.replace(",", "")  # remove thousands separators
    for pat in [r"####\s*([+-]?[\d,]+\.?\d*)", r"\\boxed\{([^}]+)\}"]:
        m = re.search(pat, text)
        if m:
            return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[+-]?\d+\.?\d*", text)
    return nums[-1] if nums else "None"


def is_correct(pred: str, true_ans: str) -> bool:
    if not pred or pred == "None":
        return False
    try:
        return abs(float(pred.replace(",", "")) - float(str(true_ans).replace(",", ""))) < 1e-6
    except (ValueError, TypeError):
        return pred.strip().upper() == str(true_ans).strip().upper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multilingual Reasoning Model")
    parser.add_argument("--lang",     type=str, default="russian")
    parser.add_argument("--baseline", type=str, default=BASELINE_MODEL,
                        help="HF Inference Providers model ID for baseline comparison")
    parser.add_argument("--samples",  type=int, default=NUM_TEST_SAMPLES)
    args = parser.parse_args()

    if not HF_TOKEN or not HF_USERNAME:
        raise EnvironmentError("HF_TOKEN and HF_USERNAME must be set in .env")

    dataset_name       = f"{HF_USERNAME}/multilingual-reasoning-{args.lang}"
    trained_model_name = f"{HF_USERNAME}/grpo-reasoning-{args.lang}"
    q_key              = f"question_{args.lang}"

    # ── 1. Load test data
    print(f"\U0001f4ca Loading test split from {dataset_name}...")
    test_data     = load_dataset(dataset_name, split="test", token=HF_TOKEN)
    test_problems = [item for item in test_data if item.get(q_key)][:args.samples]
    if not test_problems:
        raise ValueError(
            f"No test problems found with key '{q_key}' in {dataset_name}.\n"
            "Run 01_prepare_data.py --push_to_hub first."
        )
    print(f"   {len(test_problems)} problems loaded.")

    # ── 2. Evaluate GRPO-trained model (loaded from HF Hub)
    print(f"\n\U0001f9a5 Evaluating trained model: {trained_model_name}")
    # load_in_4bit=False: 4-bit QLoRA is NOT recommended for MoE models.
    # BitsandBytes does not support MoE 4-bit (per Unsloth docs Feb 2026).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = trained_model_name,
        max_seq_length = 2048,
        load_in_4bit   = False,
        fast_inference = False,
    )
    FastLanguageModel.for_inference(model)

    # Apply the same chat template used at training time
    tokenizer.chat_template = build_chat_template(SYSTEM_PROMPT, REASONING_START)

    grpo_results = []
    for item in tqdm(test_problems, desc="Evaluating GRPO model"):
        prompt   = str(item[q_key])
        true_ans = str(item["answer_number"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        text    = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs  = tokenizer([text], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens = 1024,
            do_sample      = True,
            temperature    = 0.6,
            pad_token_id   = tokenizer.eos_token_id,
        )
        # Decode only the generated portion (strip the prompt)
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text):].strip()
        pred = extract_answer(resp)
        grpo_results.append({
            "id":        item["id"],
            "source":    item["source"],
            "response":  resp,
            "predicted": pred,
            "actual":    true_ans,
            "correct":   is_correct(pred, true_ans),
        })

    # ── 3. Evaluate baseline via HF Inference Providers
    # The baseline receives the same system prompt but won't use our custom tags
    # naturally. extract_answer() has fallbacks (####, \boxed{}) for this.
    print(f"\n\U0001f310 Evaluating baseline [{args.baseline}] via HF Inference API...")
    client       = InferenceClient(api_key=HF_TOKEN)
    base_results = []

    for item in tqdm(test_problems, desc="Evaluating baseline"):
        prompt   = str(item[q_key])
        true_ans = str(item["answer_number"])
        try:
            resp = client.chat.completions.create(
                model    = args.baseline,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens  = 1024,
                temperature = 0.6,
                provider    = PROVIDER,
            )
            resp_text = resp.choices[0].message.content
        except Exception as e:
            print(f"  \u26a0\ufe0f  API error for {item['id']}: {e}")
            resp_text = ""
        pred = extract_answer(resp_text)
        base_results.append({
            "id":        item["id"],
            "source":    item["source"],
            "response":  resp_text,
            "predicted": pred,
            "actual":    true_ans,
            "correct":   is_correct(pred, true_ans),
        })

    if not grpo_results or not base_results:
        raise RuntimeError("No results to evaluate — check above errors.")

    # ── 4. Metrics
    grpo_acc = sum(r["correct"] for r in grpo_results) / len(grpo_results)
    base_acc = sum(r["correct"] for r in base_results) / len(base_results)

    print(f"\n{'='*45}")
    print(f"  GRPO Trained  : {grpo_acc:.1%}")
    print(f"  Baseline      : {base_acc:.1%}")
    print(f"  Delta         : {(grpo_acc - base_acc):+.1%}")
    print(f"  {'='*43}")
    for src in ["gsm8k", "math"]:
        g = [r for r in grpo_results if r["source"] == src]
        b = [r for r in base_results if r["source"] == src]
        if g and b:
            print(f"  {src.upper():<8}  GRPO {sum(r['correct'] for r in g)/len(g):.1%}  "
                  f"Baseline {sum(r['correct'] for r in b)/len(b):.1%}")
    print(f"{'='*45}")

    # ── 5. Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [
        f"Baseline\n({args.baseline.split('/')[-1]})",
        f"GRPO Trained\n({trained_model_name.split('/')[-1]})",
    ]
    bars = ax.bar(labels, [base_acc, grpo_acc], color=["#94a3b8", "#38bdf8"], width=0.5)
    ax.set_title(f"Qwen3-30B-A3B MoE Reasoning \u2014 {args.lang.title()}", fontsize=14)
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1.05])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    for bar, val in zip(bars, [base_acc, grpo_acc]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.02,
            f"{val:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=13,
        )
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    out_png = f"results/eval_{args.lang}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    pd.DataFrame(grpo_results).to_csv(f"results/grpo_results_{args.lang}.csv",     index=False)
    pd.DataFrame(base_results).to_csv(f"results/baseline_results_{args.lang}.csv", index=False)
    print(f"\n\u2705 Results saved to {out_png}")
