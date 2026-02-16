import torch
from vllm import LLM, SamplingParams

prompts = [
    "Reverse a string without using built-in reverse functions",
    "Check if two strings are anagrams",
    "Find the missing number in 1 to n (array contains 1..n except one number)",
    "Longest substring without repeating characters",
    "Product of array except self (no division allowed)",
    "3Sum (find all unique triplets that sum to zero)",
    "LRU Cache (implement with O(1) get and put)",
    "Number of islands (grid of '1's and '0's)",
    "Coin change (minimum number of coins needed to make amount)",
    "Trapping rain water (given height array, compute trapped water)"
]

if __name__ == "__main__":
  models = ["Qwen/Qwen3-Coder-30B-A3B-Instruct", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "openai/gpt-oss-20b"]

  for model in models:
    llm = LLM(model)
    sampling_params = SamplingParams(temperature=0.7)

    for prompt in prompts:
      output = llm.generate(
        prompt,
        sampling_params,
        use_tqdm=True,
      )
      print(output)