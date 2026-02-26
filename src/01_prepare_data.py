import argparse
import os
import random
import time
from typing import List, Dict
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import InferenceClient, login
from tqdm.auto import tqdm
from dotenv import load_dotenv

# Load env variables
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME")

# SOTA open multilingual model on HF Inference Providers
TRANSLATION_MODEL = "Qwen/Qwen3-235B-A22B-Instruct" 
PROVIDER = "auto"  

class ReasonDatasetCollector:
    def __init__(self, total_samples: int):
        self.total = total_samples

    def get_gsm8k(self, n: int) -> List[Dict]:
        print(f"Loading {n} GSM8K problems...")
        ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(n))
        return [{"id": f"gsm8k_{i}", "source": "gsm8k", "question": item["question"], 
                 "answer_full": item["answer"], 
                 "answer_number": item["answer"].split("####")[-1].strip() if "####" in item["answer"] else item["answer"]} 
                for i, item in enumerate(ds)]

    def get_math(self, n: int) -> List[Dict]:
        print(f"Loading {n} MATH problems...")
        ds = load_dataset("lighteval/MATH", "all", split="train").shuffle(seed=42).select(range(n))
        return [{"id": f"math_{i}", "source": "math", "question": item["problem"], 
                 "answer_full": item["solution"], "answer_number": item.get("answer", "")} 
                for i, item in enumerate(ds)]

    def collect(self) -> List[Dict]:
        n_gsm = int(self.total * 0.4)
        n_math = self.total - n_gsm
        data = self.get_gsm8k(n_gsm) + self.get_math(n_math)
        random.seed(42)
        random.shuffle(data)
        print(f"✅ Total collected: {len(data)} problems")
        return data

class HFTranslator:
    def __init__(self, token: str, model: str, provider: str):
        self.client = InferenceClient(api_key=token)
        self.model = model
        self.provider = provider
    
    def get_system_prompt(self, lang: str) -> str:
        prompts = {
            "russian": "You are an expert mathematical translator. Translate the problem into Russian. Use the formal, precise terminology typical of the Soviet mathematical tradition. Preserve all numbers, equations, and LaTeX formatting exactly. Output ONLY the translated text.",
            "arabic": "You are an expert mathematical translator. Translate the problem into Modern Standard Arabic. Use classical mathematical terminology. Preserve all numbers, equations, and LaTeX formatting exactly. Output ONLY the translated text."
        }
        return prompts.get(lang, f"Translate to {lang}. Preserve all math formatting exactly. Output ONLY the translated text.")

    def translate_dataset(self, dataset: List[Dict], lang: str) -> List[Dict]:
        print(f"\n🌍 Translating {len(dataset)} items to {lang.upper()} using {self.model}...")
        sys_prompt = self.get_system_prompt(lang)
        
        for item in tqdm(dataset, desc=lang):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": item["question"]}],
                    max_tokens=1500, temperature=0.2, provider=self.provider
                )
                item[f"question_{lang}"] = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Translation failed for {item['id']}: {e}")
                item[f"question_{lang}"] = None
            time.sleep(0.1) # Gentle rate limiting
            
        return [item for item in dataset if item.get(f"question_{lang}")]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual Data Preparation")
    parser.add_argument("--lang", type=str, default="russian", help="Target language")
    parser.add_argument("--samples", type=int, default=1000, help="Total number of samples")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--push_to_hub", action="store_true", help="Push final dataset to HF Hub")
    args = parser.parse_args()

    if not HF_TOKEN or not HF_USERNAME:
        raise ValueError("Please set HF_TOKEN and HF_USERNAME in .env")

    login(HF_TOKEN)
    
    collector = ReasonDatasetCollector(args.samples)
    all_problems = collector.collect()

    split_idx = int(len(all_problems) * (1 - args.test_split))
    train_data, test_data = all_problems[:split_idx], all_problems[split_idx:]

    translator = HFTranslator(HF_TOKEN, TRANSLATION_MODEL, PROVIDER)
    train_lang = translator.translate_dataset(train_data, args.lang)
    test_lang = translator.translate_dataset(test_data, args.lang)
    
    print(f"✅ Translated {len(train_lang)} train and {len(test_lang)} test records for {args.lang}.")
    
    # Convert to HF DatasetDict
    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_lang),
        "test": Dataset.from_list(test_lang)
    })

    if args.push_to_hub:
        repo_id = f"{HF_USERNAME}/multilingual-reasoning-{args.lang}"
        print(f"🚀 Pushing dataset to Hub: {repo_id}")
        ds_dict.push_to_hub(repo_id, private=False)
        print("🎉 Dataset successfully pushed!")