"""
test_get_30_tokens_vllm.py

Tests the get_30_tokens logic using vllm-served models via the OpenAI-compatible
/v1/completions endpoint with logprobs=-1 (full vocab).

Models and ports (from quant_init.sh):
  - Pradheep1647/adrl-qwen2.5-coder-4bit   -> http://localhost:8000
  - Pradheep1647/adrl-codestral-4bit        -> http://localhost:8001
  - openai/gpt-oss-20b                      -> http://localhost:8002

For each model:
  - Requests 1 token with logprobs=-1 to get the full vocab log-prob distribution.
  - Picks top-5 (highest logprob) and bottom-5 (lowest logprob) tokens.

Prints one iteration over all 30 tokens (10 per model: top5 + bottom5).
"""

import requests

# ── server config from quant_init.sh ─────────────────────────────────────────
MODELS = [
    # {"name": "Pradheep1647/adrl-codestral-4bit", "base_url": "http://100.91.18.11:8001"},
    # {"name": "Pradheep1647/adrl-qwen2.5-coder-4bit", "base_url": "http://100.18.91.11:8000"},
    {"name": "openai/gpt-oss-20b", "base_url": "http://100.73.177.66:8002"},
]

# ── prompt to test ─────────────────────────────────────────────────────────────
TEST_PROMPT = "def fibonacci(n):"


def get_next_token_logprobs_vllm(model_name: str, base_url: str, prompt: str):
    """
    Query a vllm server for the log-prob distribution over the next token.
    Uses logprobs=-1 to get the full vocab distribution.

    Returns a list of (token_str, logprob) tuples sorted descending by logprob.
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 1,          # we only want the next-token distribution
        "temperature": 0.0,       # greedy — doesn't affect logprob values
        "logprobs": 20,            # top-N logprobs (vllm OpenAI-compat API requires positive int)
        "echo": False,
    }

    resp = requests.post(f"{base_url}/v1/completions", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # vllm returns logprobs as a list of dicts per output token position
    # each dict maps token_id (int as string key) -> {"logprob": float, "decoded_token": str}
    choice = data["choices"][0]
    logprobs_obj = choice.get("logprobs", {})

    # vllm ≥ 0.4 format: top_logprobs is a list (one entry per generated token)
    top_logprobs_list = logprobs_obj.get("top_logprobs", [])

    if not top_logprobs_list:
        raise ValueError(f"No logprobs returned from {model_name}. "
                         "Make sure the vllm server is running and logprobs is supported.")

    token_logprob_map = top_logprobs_list[0]  # dict for position 0 (the one generated token)

    # Normalise to list of (token_str, logprob)
    distribution = []
    for token_id_str, info in token_logprob_map.items():
        if isinstance(info, dict):
            lp = info.get("logprob", float("-inf"))
            tok = info.get("decoded_token", info.get("token", str(token_id_str)))
        else:
            # Some vllm versions return {token_id: logprob_float} directly
            lp = float(info)
            tok = str(token_id_str)
        distribution.append((tok, lp))

    # Sort descending by logprob
    distribution.sort(key=lambda x: x[1], reverse=True)
    return distribution


def get_30_tokens_vllm(prompt: str):
    """
    For each of the 3 vllm-served models:
      - Fetch full vocab logprob distribution for the next token.
      - Extract top-5 and bottom-5 tokens.

    Returns:
      all_tokens  : list of 30 (model_name, rank_label, token, logprob) tuples
      agreement   : bool – whether all 3 models agree on the top-1 token
    """
    all_tokens = []
    top1_tokens = []

    for model in MODELS:
        distribution = get_next_token_logprobs_vllm(
            model["name"], model["base_url"], prompt
        )

        top5    = distribution[:5]
        bottom5 = distribution[-5:][::-1]  # worst → slightly less worst

        top1_tokens.append(top5[0][0])

        for rank, (tok, lp) in enumerate(top5, start=1):
            all_tokens.append((model["name"], f"top-{rank}", tok, lp))
        for rank, (tok, lp) in enumerate(bottom5, start=1):
            all_tokens.append((model["name"], f"bottom-{rank}", tok, lp))

    teachers_agreement = len(set(top1_tokens)) == 1 if len(top1_tokens) >= 2 else True
    return all_tokens, teachers_agreement


def generate_full_response(
    prompt: str,
    model_index: int = 0,
    max_tokens: int = 512,
    temperature: float = 0.0,
    stop: list = None,
) -> str:
    """
    Generate a full text completion from a vllm-served model.

    Args:
        prompt      : The input prompt string.
        model_index : Index into MODELS list (default 0).
        max_tokens  : Maximum number of tokens to generate.
        temperature : Sampling temperature (0.0 = greedy).
        stop        : Optional list of stop strings.

    Returns:
        The generated text as a string.
    """
    model = MODELS[model_index]
    payload = {
        "model": model["name"],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        payload["stop"] = stop

    resp = requests.post(f"{model['base_url']}/v1/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["text"]


def print_results(all_tokens, teachers_agreement, prompt: str):
    print("=" * 72)
    print(f"PROMPT : {prompt!r}")
    print("=" * 72)

    current_model = None
    for model_name, rank_label, token, logprob in all_tokens:
        if model_name != current_model:
            current_model = model_name
            short = model_name.split("/")[-1]
            print(f"\n── {short} ──────────────────────────────────────────")
            print(f"  {'Rank':<12}  {'Token':<25}  {'LogProb':>10}")
            print(f"  {'-'*12}  {'-'*25}  {'-'*10}")
        print(f"  {rank_label:<12}  {repr(token):<25}  {logprob:>10.4f}")

    print()
    print("=" * 72)
    print(f"Teachers agreement on top-1 token: {teachers_agreement}")
    if teachers_agreement:
        print(f"  Agreed token: {repr(all_tokens[0][2])}")
    print("=" * 72)


if __name__ == "__main__":
    print(f"Testing get_30_tokens via vllm servers (quant_init.sh)\n")

    tokens, agreement = get_30_tokens_vllm(TEST_PROMPT)
    print_results(tokens, agreement, TEST_PROMPT)

    print("\n\n── Full response ──────────────────────────────────────────────")
    print(f"PROMPT : {TEST_PROMPT!r}\n")
    response = generate_full_response(TEST_PROMPT)
    print(response)
    print("=" * 72)
