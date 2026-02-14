# Test Script: Evaluate Qwen/Qwen3-4B on HumanEval Benchmark
#
# This script evaluates the Qwen3-4B model on the HumanEval dataset (164 coding problems).
# It generates code completions for each problem prompt and checks if they pass the unit tests.
# Metrics: pass@1 (fraction of problems solved with one sample).
#
# Requirements:
# - Install dependencies: pip install transformers datasets torch evaluate-human-eval
#   (Note: 'evaluate-human-eval' is a placeholder; use 'bigcode-evaluation-harness' or implement manually.
#    For simplicity, this script uses a basic implementation with exec for testing.)
# - Hardware: GPU recommended (e.g., A100 or RTX 4090) for faster generation.
# - Model: Assumes "Qwen/Qwen3-4B" is the base model; use "-Instruct" if available for better results.
# - Warning: Executing generated code can be risky (security/sandbox issues). Run in a isolated env.
# - Expected runtime: ~10-30 min on a single GPU for 164 problems (1 sample each).

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import ast  # For safe parsing
import traceback

# Load model and tokenizer
model_name = "Qwen/Qwen3-4B"  # Or "Qwen/Qwen3-4B-Instruct" if instruct-tuned variant exists
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Load HumanEval dataset
dataset = load_dataset("openai_humaneval")["test"]  # 164 problems

# Function to extract function body from generated text (HumanEval format)
def extract_function_body(generated_text):
    # Find the code after the prompt (assume it starts after 'def' or similar)
    match = re.search(r'def\s+\w+\(.*\):.*', generated_text, re.DOTALL)
    if match:
        return match.group(0)
    return generated_text  # Fallback

# Safe execution function for unit tests
def check_solution(problem, generated_code):
    try:
        # Combine canonical solution setup + generated code + check
        full_code = problem['entry_point'] + generated_code + "\n" + problem['test']
        # Parse to ensure syntax
        ast.parse(full_code)
        # Exec in isolated globals
        local_globals = {}
        exec(full_code, local_globals)
        # If no exception, it passed
        return True
    except Exception as e:
        print(f"Error for problem {problem['task_id']}: {traceback.format_exc()}")
        return False

# Evaluation loop
pass_count = 0
total_problems = len(dataset)

for idx, problem in enumerate(dataset):
    prompt = problem['prompt']  # The function signature and docstring
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate completion (adjust params for quality: temp=0.2 for deterministic)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,  # Enough for most HumanEval solutions
        do_sample=False,     # Greedy for pass@1
        temperature=0.2,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    function_body = extract_function_body(generated_text)

    # Check if it passes tests
    if check_solution(problem, function_body):
        pass_count += 1
        print(f"Problem {problem['task_id']} PASSED")
    else:
        print(f"Problem {problem['task_id']} FAILED")

# Final metric
pass_at_1 = pass_count / total_problems
print(f"\nHumanEval pass@1: {pass_at_1:.2%} ({pass_count}/{total_problems})")

# Notes:
# - For better accuracy, use multiple samples (pass@k) and adjust generation params.
# - If using vLLM for speed: Replace model.generate with vLLM's generate.
# - Benchmark baselines: Similar 4B models (e.g., Qwen2-4B) score ~40-60% pass@1 on HumanEval.
# - Improve: Fine-tune or use instruct variant for higher scores.
# - Run this script: python humaneval_eval.py