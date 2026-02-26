# Multilingual Reasoning GRPO

This repository contains a full end-to-end pipeline for training and evaluating multilingual reasoning models using GRPO (Group Relative Policy Optimization). 

It is designed for high-end hardware (e.g., NVIDIA H100) and integrates deeply with the Hugging Face ecosystem (Datasets Hub, Inference Providers, Model Hub).

## Architecture
- **Data Prep**: Collects English math/science problems (GSM8K, MATH, ARC), translates them via Hugging Face Inference Providers (using Qwen3-235B), and pushes the datasets to your Hugging Face account.
- **Training**: Uses `unsloth` and `trl` to run GRPO on an H100 GPU. Optimizations include `bfloat16` precision (no 4-bit quantization), vLLM for fast generation, larger sequence lengths (`2048`), and `NUM_GENERATIONS=8`.
- **Evaluation**: Compares the GRPO-trained model (loaded locally from HF Hub) against a baseline via the HF Inference API.

## Setup
1. Clone the repository.
2. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your Hugging Face token and username.

## Usage
1. **Prepare Data**: 
   Collects datasets, translates to the target language, and pushes the generated dataset to `hf.co/<your_username>/multilingual-reasoning-<lang>`.
   ```bash
   python src/01_prepare_data.py --lang russian --push_to_hub
   ```

2. **Train Model**: 
   Pulls your dataset from the Hub, loads the model in pure `bfloat16`, trains with GRPO on H100, and merges/pushes the final model to your HF account.
   ```bash
   python src/02_train_grpo.py --lang russian
   ```

3. **Evaluate**: 
   Evaluates your model vs a baseline and saves the resulting CSV/PNG to `results/`.
   ```bash
   python src/03_evaluate.py --lang russian
   ```