"""
Test for get_30_tokens() in structs/main.py

Uses a single Qwen/Qwen3-4B model loaded three times via SGLang to simulate
three teacher models. We do this to avoid needing three separate GPU allocations.

What is tested:
1. Return value is a list of length 2: [token_list, teachers_agreement]
2. token_list has exactly 30 entries (top5 + bottom5) * 3 models
3. Each entry is a 3-tuple: (token_id: int, decoded_text: str, logprob: float)
4. top5 entries per model are sorted descending by logprob (highest first)
5. bottom5 entries per model are sorted ascending by logprob (lowest first)
6. teachers_agreement is a bool
7. teachers_agreement is True when all three models (same model here) agree on top-1 token
"""

import sys
import os

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sglang as sgl
from transformers import AutoTokenizer, AutoModelForCausalLM
from structs.main import get_30_tokens

if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen3-4B"
    TEST_PROMPT = "def add(a, b):"

    print("Loading model (shared weights, single instance)...")
    # Use a small mem_fraction_static so the engine fits in available GPU memory.
    hf_tm1 = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", device_map="auto", trust_remote_code=True)
    hf_tm2 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct", device_map="auto", trust_remote_code=True)
    hf_tm3 = AutoModelForCausalLM.from_pretrained("mistralai/Codestral-22B-v0.1", device_map="auto", trust_remote_code=True)

    tokenzer1 = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    tokenzer2 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
    tokenzer3 = AutoTokenizer.from_pretrained("mistralai/Codestral-22B-v0.1")

    print(f"gpt Vocab size: {tokenzer1.vocab_size}")
    print(f"Qwen Vocab size: {tokenzer2.vocab_size}")
    print(f"DeepSeek Vocab size: {tokenzer3.vocab_size}")
    print(f"Running get_30_tokens with prompt: {repr(TEST_PROMPT)}\n")

    result = get_30_tokens(hf_tm1, hf_tm2, hf_tm3, TEST_PROMPT, tokenzer1, tokenzer2, tokenzer3)

    # ── Test 1: return value structure ──────────────────────────────────────────
    assert isinstance(result, list) and len(result) == 2, \
        f"Expected list of length 2, got {type(result)} len={len(result)}"
    print("PASS  Test 1: return is [token_list, teachers_agreement]")

    token_list, teachers_agreement = result

    # ── Test 2: token_list length ────────────────────────────────────────────────
    assert len(token_list) == 30, \
        f"Expected 30 tokens, got {len(token_list)}"
    print("PASS  Test 2: token_list has 30 entries")

    # ── Test 3: each entry is a 3-tuple (int, str, float) ────────────────────────
    for idx, entry in enumerate(token_list):
        assert isinstance(entry, tuple) and len(entry) == 3, \
            f"Entry {idx} is not a 3-tuple: {entry}"
        t_id, decoded, lp = entry
        assert isinstance(t_id, int), f"Entry {idx}: token_id not int ({type(t_id)})"
        assert isinstance(decoded, str), f"Entry {idx}: decoded not str ({type(decoded)})"
        assert isinstance(lp, float), f"Entry {idx}: logprob not float ({type(lp)})"
    print("PASS  Test 3: all entries are (int, str, float) tuples")

    # ── Test 4: top5 blocks are descending (highest logprob first) ────────────────
    # Layout: [top5_1(0-4), bot5_1(5-9), top5_2(10-14), bot5_2(15-19), top5_3(20-24), bot5_3(25-29)]
    for model_idx, top_start in enumerate([0, 10, 20]):
        top5 = token_list[top_start : top_start + 5]
        logprobs = [entry[2] for entry in top5]
        assert logprobs == sorted(logprobs, reverse=True), \
            f"Model {model_idx+1} top5 not sorted descending: {logprobs}"
    print("PASS  Test 4: top5 blocks are sorted descending by logprob")

    # ── Test 5: bottom5 blocks are ascending (lowest logprob first) ───────────────
    for model_idx, bot_start in enumerate([5, 15, 25]):
        bot5 = token_list[bot_start : bot_start + 5]
        logprobs = [entry[2] for entry in bot5]
        assert logprobs == sorted(logprobs), \
            f"Model {model_idx+1} bottom5 not sorted ascending: {logprobs}"
    print("PASS  Test 5: bottom5 blocks are sorted ascending by logprob")

    # ── Test 6: teachers_agreement is a bool ─────────────────────────────────────
    assert isinstance(teachers_agreement, bool), \
        f"teachers_agreement is not bool: {type(teachers_agreement)}"
    print("PASS  Test 6: teachers_agreement is a bool")

    # ── Test 7: agreement value is correct (same model → should always agree) ────
    top1_model1 = token_list[0][0]   # top5_1[0] token_id
    top1_model2 = token_list[10][0]  # top5_2[0] token_id
    top1_model3 = token_list[20][0]  # top5_3[0] token_id
    expected_agreement = (top1_model1 == top1_model2 == top1_model3)
    assert teachers_agreement == expected_agreement, \
        f"teachers_agreement mismatch: got {teachers_agreement}, expected {expected_agreement}"
    # With the same model, top-1 must always match → True
    assert teachers_agreement is True, \
        f"Same model used three times should always agree, got {teachers_agreement}"
    print("PASS  Test 7: teachers_agreement is True (same model used for all three)")

    # ── Summary ───────────────────────────────────────────────────────────────────
    print("\nAll 7 tests passed.")
    print(f"\nSample output (first entry of each top5 block):")
    for model_idx, top_start in enumerate([0, 10, 20]):
        entry = token_list[top_start]
        print(f"  Model {model_idx+1} top-1 token: id={entry[0]}, text={repr(entry[1])}, logprob={entry[2]:.4f}")
    print(f"  teachers_agreement: {teachers_agreement}")
