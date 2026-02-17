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
  models = ["bigcode/starcoder2-15b", "Qwen/Qwen2.5-Coder-14B-Instruct", "ServiceNow-AI/Apriel-1.5-15b-Thinker"]

  for model in models:
    llm = LLM(model, gpu_memory_utilization=0.3)
    sampling_params = SamplingParams(temperature=0.7)

    for prompt in prompts:
      output = llm.generate(
        prompt,
        sampling_params,
        use_tqdm=True,
      )
      print(output[0].outputs[0].text)
