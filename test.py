import torch
from unsloth import FastLanguageModel
from vllm import LLM, SamplingParams

prompts = [
    "reverse a string without using built-in reverse functions",
    "check if two strings are anagrams",
    "find the missing number in 1 to n (array contains 1..n except one number)",
    "longest substring without repeating characters",
    "product of array except self (no division allowed)",
    "3sum (find all unique triplets that sum to zero)",
    "lru cache (implement with o(1) get and put)",
    "number of islands (grid of '1's and '0's)",
    "coin change (minimum number of coins needed to make amount)",
    "trapping rain water (given height array, compute trapped water)"
]

if __name__ == "__main__":
    models = [
        "bigcode/starcoder2-15b",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "ServiceNow-AI/Apriel-1.5-15b-Thinker"
    ]

    for model_name in models:
        llm = LLM(model_name, max_model_len=4096)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

        for prompt in prompts:
            output = llm.generate(prompt, sampling_params)
            print(f"prompt: {prompt}, resp: {output}, model: {model_name}")

    torch.cuda.empty_cache()
