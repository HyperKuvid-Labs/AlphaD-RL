import torch
from unsloth import FastLanguageModel

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
        print(f"\n{'='*80}")
        print(f"loading {model_name}")
        print(f"{'='*80}\n")

        try:
            # full precision cuz h200 go brrr
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name         = model_name,
                max_seq_length     = 4096,
                dtype              = None,
                load_in_4bit       = False,
            )

            FastLanguageModel.for_inference(model)

            for prompt in prompts:
                print(f"\nprompt: {prompt}")
                print("-" * 60)

                full_prompt = f"write clean efficient python code for this:\n{prompt}\n\n```python\n"

                inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens  = 1200,
                        temperature     = 0.7,
                        top_p           = 0.95,
                        repetition_penalty = 1.1,
                        use_cache       = True,
                    )

                response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

                if "```python" in response:
                    code = response.split("```python", 1)[-1].split("```", 1)[0].strip()
                    print(code)
                else:
                    print(response[len(full_prompt):].strip())

                print("-" * 60)

        except Exception as e:
            print(f"failed on {model_name}")
            print(str(e))
            print("skipping...\n")

        torch.cuda.empty_cache()
