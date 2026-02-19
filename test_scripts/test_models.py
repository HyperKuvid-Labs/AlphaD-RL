import torch
# from unsloth import FastLanguageModel
import sglang as sgl

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
        "openai/gpt-oss-20b",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "DeepSeek-Coder-V2-Lite-Instruct"
    ]

    for model_name in models:
        llm = sgl.Engine(model_path=model_name, mem_fraction_static=0.25, context_length=4096)
        sampling_params = {"temperature": 0.5}

        for prompt in prompts:
            output = llm.generate(prompt, sampling_params)
            print(f"prompt: {prompt}, resp: {output['text']}, model: {model_name}")

        llm.shutdown()

    torch.cuda.empty_cache()
