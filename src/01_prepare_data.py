# ==============================================================================
# 01_prepare_data.py — Dataset collection, translation, and HF Hub upload
# Run: python src/01_prepare_data.py --lang russian --push_to_hub
# ==============================================================================

import argparse
import os
import random
import time
from typing import List, Dict
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import InferenceClient
from tqdm.auto import tqdm
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOTAL_SAMPLES, TEST_SPLIT, TRANSLATION_MODEL, PROVIDER

load_dotenv()
HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]


class ReasonDatasetCollector:
    def __init__(self, total_samples: int):
        self.total = total_samples

    def get_gsm8k(self, n: int) -> List[Dict]:
        print(f"Loading {n} GSM8K problems...")
        ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(n))
        return [
            {
                "id":            f"gsm8k_{i}",
                "source":        "gsm8k",
                "question":      item["question"],
                "answer_full":   item["answer"],
                "answer_number": item["answer"].split("####")[-1].strip()
                    if "####" in item["answer"] else item["answer"].strip(),
            }
            for i, item in enumerate(ds)
        ]

    def get_math(self, n: int) -> List[Dict]:
        print(f"Loading {n} MATH problems...")
        ds = load_dataset("lighteval/MATH", "all", split="train").shuffle(seed=42).select(range(n))
        problems = []
        for i, item in enumerate(ds):
            answer_number = str(item.get("answer", "")).strip()
            # Skip MATH items with no numeric answer — GRPO reward requires a ground truth
            if not answer_number:
                continue
            problems.append({
                "id":            f"math_{i}",
                "source":        "math",
                "question":      item["problem"],
                "answer_full":   item["solution"],
                "answer_number": answer_number,
            })
        return problems

    def collect(self) -> List[Dict]:
        n_gsm  = int(self.total * 0.4)
        n_math = self.total - n_gsm
        data   = self.get_gsm8k(n_gsm) + self.get_math(n_math)
        random.seed(42)
        random.shuffle(data)
        print(f"\u2705 Collected {len(data)} problems total ({n_gsm} GSM8K + up to {n_math} MATH).")
        return data


class HFTranslator:
    def __init__(self, token: str, model: str, provider: str):
        self.client   = InferenceClient(api_key=token)
        self.model    = model
        self.provider = provider

    def _system_prompt(self, lang: str) -> str:
        prompts = {
            "russian": (
                "You are an expert mathematical translator. Translate the problem into Russian. "
                "Use the formal, precise terminology of the Soviet mathematical tradition. "
                "Preserve all numbers, equations, and LaTeX formatting exactly. "
                "Output ONLY the translated text, nothing else."
            ),
            "arabic": (
                "You are an expert mathematical translator. Translate the problem into Modern Standard Arabic. "
                "Use classical mathematical terminology. "
                "Preserve all numbers, equations, and LaTeX formatting exactly. "
                "Output ONLY the translated text, nothing else."
            ),
        }
        return prompts.get(
            lang,
            f"Translate the following into {lang}. Preserve all mathematical formatting. Output ONLY the translated text."
        )

    def _translate_one(self, item: Dict, lang: str, sys_prompt: str, retries: int = 3) -> Dict:
        """Translate a single item with exponential-backoff retry."""
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model    = self.model,
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user",   "content": item["question"]},
                    ],
                    max_tokens  = 1500,
                    temperature = 0.2,
                    provider    = self.provider,
                )
                item[f"question_{lang}"] = response.choices[0].message.content.strip()
                return item
            except Exception as e:
                wait = 2 ** attempt
                print(f"  Attempt {attempt+1} failed for {item['id']}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        item[f"question_{lang}"] = None
        return item

    def translate_dataset(self, dataset: List[Dict], lang: str) -> List[Dict]:
        print(f"\n\U0001f30d Translating {len(dataset)} items \u2192 {lang.upper()} using {self.model}")
        sys_prompt = self._system_prompt(lang)
        translated = [self._translate_one(item, lang, sys_prompt) for item in tqdm(dataset, desc=lang)]

        failed  = [it for it in translated if it.get(f"question_{lang}") is None]
        success = [it for it in translated if it.get(f"question_{lang}") is not None]
        if failed:
            print(f"  \u26a0\ufe0f  {len(failed)} items failed translation and were dropped: {[f['id'] for f in failed]}")
        print(f"  \u2705 {len(success)}/{len(dataset)} successfully translated.")
        return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual Data Preparation")
    parser.add_argument("--lang",        type=str,   default="russian",         help="Target language")
    parser.add_argument("--samples",     type=int,   default=TOTAL_SAMPLES,     help="Total problems to collect")
    parser.add_argument("--test_split",  type=float, default=TEST_SPLIT,        help="Fraction for test set")
    parser.add_argument("--model",       type=str,   default=TRANSLATION_MODEL, help="HF Inference model for translation")
    parser.add_argument("--provider",    type=str,   default=PROVIDER,          help="HF Inference Provider")
    parser.add_argument("--push_to_hub", action="store_true",                   help="Push dataset to HF Hub")
    args = parser.parse_args()

    if not HF_TOKEN or not HF_USERNAME:
        raise EnvironmentError("HF_TOKEN and HF_USERNAME must be set in .env")

    # 1. Collect
    all_problems = ReasonDatasetCollector(args.samples).collect()

    # 2. Translate first, THEN split — so test proportions are accurate
    translator     = HFTranslator(HF_TOKEN, args.model, args.provider)
    all_translated = translator.translate_dataset(all_problems, args.lang)

    if not all_translated:
        raise RuntimeError("All translations failed. Check your HF_TOKEN and provider settings.")

    # 3. Split
    random.seed(42)
    random.shuffle(all_translated)
    split_idx  = int(len(all_translated) * (1 - args.test_split))
    train_data = all_translated[:split_idx]
    test_data  = all_translated[split_idx:]
    print(f"Split \u2192 train: {len(train_data)}, test: {len(test_data)}")

    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test":  Dataset.from_list(test_data),
    })

    if args.push_to_hub:
        repo_id = f"{HF_USERNAME}/multilingual-reasoning-{args.lang}"
        print(f"\U0001f680 Pushing to Hub: {repo_id}")
        ds_dict.push_to_hub(repo_id, private=False, token=HF_TOKEN)
        print(f"\U0001f389 Dataset live at https://huggingface.co/datasets/{repo_id}")
    else:
        print("Dataset ready (use --push_to_hub to upload to HF Hub).")
