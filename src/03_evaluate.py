import argparse
import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from huggingface_hub import InferenceClient
from unsloth import FastLanguageModel
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME")
PROVIDER = "auto"
NUM_TEST_SAMPLES = 50
BASELINE_MODEL = "Qwen/Qwen3.5-27B-Instruct" 

def extract_answer(text: str) -> str:
    if "<answer>" in text:
        ans = text.split("<answer>")[-1]
        return ans.split("</answer>")[0].strip() if "</answer>" in ans else ans.strip()
    for pat in [r"####\s*([+-]?\d+\.?\d*)", r"\\boxed{([^}]+)}"]:
        m = re.search(pat, text)
        if m: return m.group(1).strip()
    nums = re.findall(r"([+-]?\d+\.?\d*)", text)
    return nums[-1] if nums else "None"

def is_correct(pred: str, true_ans: str) -> bool:
    if pred == "None": return False
    try: return abs(float(pred) - float(true_ans)) < 1e-6
    except: return pred.strip().upper() == str(true_ans).strip().upper()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multilingual Reasoning Model")
    parser.add_argument("--lang", type=str, default="russian", help="Target language")
    args = parser.parse_args()

    dataset_name = f"{HF_USERNAME}/multilingual-reasoning-{args.lang}"
    trained_model_name = f"{HF_USERNAME}/grpo-reasoning-{args.lang}"

    print(f"📊 Loading test data from {dataset_name}...")
    test_data = load_dataset(dataset_name, split="test")
    q_key = f"question_{args.lang}"
    test_problems = [item for item in test_data if item.get(q_key)][:NUM_TEST_SAMPLES]

    print(f"🦥 Evaluating {trained_model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=trained_model_name,
        max_seq_length=2048,
        load_in_4bit=True # 4-bit is fine for fast inference evaluation
    )
    FastLanguageModel.for_inference(model)

    grpo_results = []
    sys_prompt = "Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
    
    for item in tqdm(test_problems, desc="GRPO"):
        prompt = item[q_key]
        true_ans = item["answer_number"]
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.6)
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(text):].strip()
        
        pred = extract_answer(resp)
        grpo_results.append({
            "id": item["id"], "source": item["source"],
            "response": resp, "predicted": pred, "actual": true_ans,
            "correct": is_correct(pred, true_ans)
        })

    print(f"🌐 Evaluating Baseline {BASELINE_MODEL} via HF API...")
    client = InferenceClient(api_key=HF_TOKEN)
    base_results = []

    for item in tqdm(test_problems, desc="Baseline"):
        prompt = item[q_key]
        true_ans = item["answer_number"]
        try:
            resp = client.chat.completions.create(
                model=BASELINE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024, temperature=0.6, provider=PROVIDER
            )
            resp_text = resp.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            resp_text = ""
            
        pred = extract_answer(resp_text)
        base_results.append({
            "id": item["id"], "source": item["source"],
            "response": resp_text, "predicted": pred, "actual": true_ans,
            "correct": is_correct(pred, true_ans)
        })

    grpo_acc = sum(r["correct"] for r in grpo_results) / len(grpo_results)
    base_acc = sum(r["correct"] for r in base_results) / len(base_results)

    print(f"\n🎯 GRPO Accuracy: {grpo_acc:.1%}")
    print(f"🎯 Baseline Accuracy: {base_acc:.1%}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Baseline', 'GRPO Trained'], [base_acc, grpo_acc], color=['#94a3b8', '#38bdf8'])
    ax.set_title(f'Qwen3.5 Reasoning ({args.lang.title()})')
    ax.set_ylim([0, 1.0])
    for i, v in enumerate([base_acc, grpo_acc]):
        ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
    
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/eval_{args.lang}.png")
    pd.DataFrame(grpo_results).to_csv(f"results/grpo_results_{args.lang}.csv", index=False)
    print("✅ Evaluation complete and results saved to `results/`.")