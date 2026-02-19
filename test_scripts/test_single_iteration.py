"""
Test for a single MCTS iteration + level_guesser flow in structs/main.py

Simulates ONE complete iteration of the training pipeline WITHOUT running GRPO:

  Stage 1 – Golden solution
      generate_best_solution() is called so the teachers propose and score
      solutions to the test prompt.  We wrap each HF teacher model with a thin
      HFEngineWrapper that mirrors the sgl.Engine .generate() interface.

  Stage 2 – MCTS expansion
      A root Node is created, then expand_leaf() is called once.
      Internally this calls get_30_tokens() with three HF teacher models.

  Stage 3 – Process reward
      get_process_reward() asks the teachers to score the partial solution
      generated so far (the root node has an empty generated_text, so we
      use the top-1 token text from the expansion as a minimal partial).

  Stage 4 – Back-propagation
      backpropagte() is called and the root node's statistics are updated.

  Stage 5 – Level-guesser decision
      level_guesser() runs Qwen3-4B (the student model) and returns a bool
      indicating whether MCTS should stop early.

Teacher models (mirrors the production config in structs/main.py):
  tm1  – openai/gpt-oss-20b
  tm2  – Qwen/Qwen2.5-Coder-14B-Instruct
  tm3  – deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct

Student / level-guesser model:
  Qwen/Qwen3-4B  (used ONLY in Stage 5)

The HFEngineWrapper converts the sgl.Engine call-signature
``engine.generate([prompt], params)`` into HF generation so that
generate_best_solution() and get_process_reward() work unchanged.

What is tested:
  1. HFEngineWrapper returns a list whose first element has a non-empty 'text' key
  2. generate_best_solution() returns a non-empty Python string
  3. expand_leaf() returns a bool (teachers_agreement) and populates children
  4. Root node has 30 children after expansion
  5. Each child node has a non-empty generated_text
  6. get_process_reward() returns a float in [-1.0, 1.0]
  7. After backpropagte(), root.visit_count == 1 and root.value == reward
  8. level_guesser() returns a bool
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer, AutoModelForCausalLM
from structs.main import (
    Node,
    expand_leaf,
    get_process_reward,
    generate_best_solution,
    backpropagte,
    level_guesser,
    get_next_token_logprobs_hf,
)


# ---------------------------------------------------------------------------
# Thin wrapper that makes an HF model look like a sgl.Engine so that
# generate_best_solution() and get_process_reward() work without change.
# ---------------------------------------------------------------------------
class HFEngineWrapper:
    """Mimics the sgl.Engine interface used in generate_best_solution /
    get_process_reward:  engine.generate([prompt_str, ...], params)
    returns  [{'text': generated_text}, ...]
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompts: list, params: dict) -> list:
        results = []
        max_new_tokens = params.get("max_new_tokens", 128)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 1.0)

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            # enable_thinking=False suppresses Qwen3's <think> block; other
            # models that don't support the kwarg fall back gracefully.
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = self.tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.device)

            do_sample = temperature > 0.0
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    do_sample=do_sample,
                    use_cache=True,
                )

            # Only return the newly generated tokens (skip the prompt)
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append({"text": text})

        return results


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Teacher model names – same as the production config in structs/main.py
    TEACHER_MODEL_1 = "openai/gpt-oss-20b"
    TEACHER_MODEL_2 = "Qwen/Qwen2.5-Coder-14B-Instruct"
    TEACHER_MODEL_3 = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    # Student / level-guesser model
    STUDENT_MODEL   = "Qwen/Qwen3-4B"

    TEST_PROMPT = (
        "Write a Python function `def add(a, b):` that returns the sum of a and b."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load teacher models ──────────────────────────────────────────────────
    print(f"\nLoading teacher model 1: {TEACHER_MODEL_1} ...")
    hf_tm1 = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_1, device_map="auto", trust_remote_code=True
    )
    tok1 = AutoTokenizer.from_pretrained(TEACHER_MODEL_1, trust_remote_code=True)
    tok1.pad_token = tok1.pad_token or tok1.eos_token

    print(f"Loading teacher model 2: {TEACHER_MODEL_2} ...")
    hf_tm2 = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_2, device_map="auto", trust_remote_code=True
    )
    tok2 = AutoTokenizer.from_pretrained(TEACHER_MODEL_2, trust_remote_code=True)
    tok2.pad_token = tok2.pad_token or tok2.eos_token

    print(f"Loading teacher model 3: {TEACHER_MODEL_3} ...")
    hf_tm3 = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_3, device_map="auto", trust_remote_code=True
    )
    tok3 = AutoTokenizer.from_pretrained(TEACHER_MODEL_3, trust_remote_code=True)
    tok3.pad_token = tok3.pad_token or tok3.eos_token

    # sgl.Engine-compatible wrappers, one per teacher
    tm1 = HFEngineWrapper(hf_tm1, tok1, device=device)
    tm2 = HFEngineWrapper(hf_tm2, tok2, device=device)
    tm3 = HFEngineWrapper(hf_tm3, tok3, device=device)

    # ── Load student model (level_guesser only) ──────────────────────────────
    print(f"\nLoading student model (level_guesser): {STUDENT_MODEL} ...")
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, device_map="auto", trust_remote_code=True
    )
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    student_tokenizer.pad_token = student_tokenizer.eos_token

    params = {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 128}

    # ── Stage 1: golden solution ─────────────────────────────────────────────
    print("\n[Stage 1] generate_best_solution ...")
    golden_solution = generate_best_solution(
        TEST_PROMPT, tm1, tm2, tm3, params, params, params
    )

    # Test 1: wrapper returns correct structure (use tm1 as representative)
    raw_out = tm1.generate([TEST_PROMPT], {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 32})
    assert isinstance(raw_out, list) and len(raw_out) == 1 and "text" in raw_out[0], \
        f"HFEngineWrapper output malformed: {raw_out}"
    assert isinstance(raw_out[0]["text"], str) and len(raw_out[0]["text"]) > 0, \
        "HFEngineWrapper returned empty text"
    print("PASS  Test 1: HFEngineWrapper.generate() returns [{'text': str}]")

    # Test 2: generate_best_solution returns a non-empty string
    assert isinstance(golden_solution, str) and len(golden_solution.strip()) > 0, \
        f"generate_best_solution returned empty or non-string: {repr(golden_solution)}"
    print("PASS  Test 2: generate_best_solution() returned a non-empty string")
    print(f"        Golden solution preview: {repr(golden_solution[:120])} ...")

    # ── Stage 2: MCTS expansion ──────────────────────────────────────────────
    print("\n[Stage 2] MCTS expansion (expand_leaf) ...")
    root_node = Node(token_id=None, generated_text="", parent=None)

    teachers_agreement = expand_leaf(
        root_node, TEST_PROMPT, hf_tm1, hf_tm2, hf_tm3, tok1, tok2, tok3
    )

    # Test 3: expand_leaf returns a bool
    assert isinstance(teachers_agreement, bool), \
        f"expand_leaf should return bool, got {type(teachers_agreement)}"
    print(f"PASS  Test 3: expand_leaf() returned bool (teachers_agreement={teachers_agreement})")

    # Test 4: root node has exactly 30 children  (top5 + bottom5) × 3 models
    assert len(root_node.children) == 30, \
        f"Expected 30 children after expansion, got {len(root_node.children)}"
    print("PASS  Test 4: root_node has exactly 30 children after expansion")

    # Test 5: every child has non-empty generated_text
    for idx, child in enumerate(root_node.children):
        assert isinstance(child.generated_text, str) and len(child.generated_text) > 0, \
            f"Child {idx} has empty or non-string generated_text: {repr(child.generated_text)}"
    print("PASS  Test 5: all 30 child nodes have non-empty generated_text")

    first_child_text = root_node.children[0].generated_text
    print(f"        First child token text: {repr(first_child_text)}")

    # ── Stage 3: process reward ──────────────────────────────────────────────
    print("\n[Stage 3] get_process_reward ...")
    partial_solution = first_child_text   # minimal partial: one token

    reward = get_process_reward(
        TEST_PROMPT,
        golden_solution,
        partial_solution,
        tm1, tm2, tm3,
        tok1, tok2, tok3,
    )

    # Test 6: reward is a float in [-1.0, 1.0]
    assert isinstance(reward, float), \
        f"get_process_reward should return float, got {type(reward)}"
    assert -1.0 <= reward <= 1.0, \
        f"Reward out of expected range [-1, 1]: {reward}"
    print(f"PASS  Test 6: get_process_reward() returned float in [-1, 1]: {reward:.4f}")

    # ── Stage 4: back-propagation ────────────────────────────────────────────
    print("\n[Stage 4] backpropagte ...")
    backpropagte(root_node.children[0], reward)

    # Test 7: visit_count and value updated on leaf and root
    leaf = root_node.children[0]
    assert leaf.visit_count == 1, \
        f"Leaf visit_count should be 1 after one backprop, got {leaf.visit_count}"
    assert leaf.value == reward, \
        f"Leaf value should equal reward {reward}, got {leaf.value}"
    assert root_node.visit_count == 1, \
        f"Root visit_count should be 1 after backprop, got {root_node.visit_count}"
    assert root_node.value == reward, \
        f"Root value should equal reward {reward}, got {root_node.value}"
    print(f"PASS  Test 7: backpropagte() updated leaf and root (visit=1, value={reward:.4f})")

    # ── Stage 5: level-guesser decision ─────────────────────────────────────
    print("\n[Stage 5] level_guesser (student model = Qwen3-4B) ...")

    seq_length   = len(leaf.generated_text)
    node_count   = len(root_node.children)
    avg_value    = sum(c.value for c in root_node.children) / node_count

    should_stop = level_guesser(
        student_model, student_tokenizer, seq_length, teachers_agreement, avg_value, node_count
    )

    # Test 8: level_guesser returns a bool
    assert isinstance(should_stop, bool), \
        f"level_guesser should return bool, got {type(should_stop)}"
    print(f"PASS  Test 8: level_guesser() returned bool: should_stop={should_stop}")
    print(f"        Inputs → seq_length={seq_length}, agreement={teachers_agreement}, "
          f"avg_value={avg_value:.4f}, node_count={node_count}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("All 8 tests passed — single MCTS iteration flow is working.")
    print("="*60)
    print(f"\nFull iteration summary:")
    print(f"  Prompt            : {repr(TEST_PROMPT[:60])} ...")
    print(f"  Golden solution   : {repr(golden_solution[:80])} ...")
    print(f"  Teacher models    : {TEACHER_MODEL_1}, {TEACHER_MODEL_2}, {TEACHER_MODEL_3}")
    print(f"  Student model     : {STUDENT_MODEL}")
    print(f"  Teachers agree?   : {teachers_agreement}")
    print(f"  Process reward    : {reward:.4f}")
    print(f"  should_stop       : {should_stop}")
