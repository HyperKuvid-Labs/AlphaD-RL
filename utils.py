import requests

# ── teacher backend registry ─────────────────────────────────────────────────
# maps each teacher model id → vllm server base url.
# long-form generations (best solutions, continuations, process-reward scoring)
# are routed to these hosted vllm endpoints via http.
# token-level logprob queries (expand_leaf / get_next_token_logprobs_hf) still
# run on the locally-loaded hf models — they need access to the raw logits.
TEACHER_ENDPOINTS: dict = {
    "openai/gpt-oss-20b":                          "http://100.109.49.52:8000",
    "Qwen/Qwen2.5-Coder-14B-Instruct":             "http://100.82.195.32:8000",
    "mistralai/Codestral-22B-v0.1": "http://100.121.167.86:8000",
}

# Ordered list: index 0 = tm1 (GPT), 1 = tm2 (Qwen), 2 = tm3 (DeepSeek)
TEACHER_MODEL_IDS: list = list(TEACHER_ENDPOINTS.keys())

def _vllm_generate(model_id: str, prompts: list, params: dict) -> list:
    """
    sends long-form generation requests to a hosted vllm server.

    calls POST <base_url>/v1/completions for each prompt and returns a list of dicts
    with key 'text', matching the interface of _hf_generate so call-sites are
    interchangeable.

    use this for:
      - generate_best_solution   (up to 1024 tokens per teacher)
      - continuation completions in _terminate_and_evaluate
      - get_process_reward scoring
    do NOT use this for get_next_token_logprobs_hf / expand_leaf — those need
    raw logit tensors and must stay on the local hf model.
    """
    base_url = TEACHER_ENDPOINTS[model_id]
    results = []
    for prompt in prompts:
        payload = {
            "prompt": prompt,
            "max_tokens": params.get("max_new_tokens", 1024),
            "temperature": params.get("temperature", 0.7),
        }
        try:
            resp = requests.post(f"{base_url}/v1/completions", json=payload, timeout=120)
            resp.raise_for_status()
            results.append({"text": resp.json().get("choices", [{}])[0].get("text", "")})
        except Exception as e:
            print(f"[WARNING] _vllm_generate({model_id}) failed: {e}")
            results.append({"text": ""})
    return results


def _hf_generate(model, tokenizer, prompts, params):
    """
    runs hf model.generate() for a list of prompts.
    returns a list of dicts with key 'text' (newly generated tokens only),
    matching the interface of _vllm_generate so call-sites are interchangeable.
    """
    import torch
    device = next(model.parameters()).device
    do_sample = params.get("temperature", 0.7) > 0
    results = []
    for prompt in prompts:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)
        with torch.no_grad():
            try:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=params.get("max_new_tokens", 512),
                    temperature=params.get("temperature", 0.7),
                    top_p=params.get("top_p", 1.0),
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    use_cache=False,
                )
            except AttributeError as e:
                print(f"[WARNING] _hf_generate sampling failed ({e}), falling back to greedy")
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=params.get("max_new_tokens", 512),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    use_cache=False,
                )
        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append({"text": text})
    return results


def extract_code(code):
  import re
  code = re.sub(r'```(?:python)?\n', '', code)
  code = re.sub(r'```', '', code)
  return code.strip()

def get_best_time_complexity(output1, output2, output3):
   # my idea is that the time complexity will be printedin terms of n, if i have some number like 100, i can compatre between them and return the choice like 1, 2, 3 accordingly
  import re
  complexities = []
  for output in [output1, output2, output3]:
      match = re.search(r'Time Complexity:\s*O\(([^)]+)\)', output)
      if match:
          complexity = match.group(1).strip()
          complexities.append(complexity)
      else:
          complexities.append(None)

  values = []
  for c in complexities:
    if c is not None:
       # need to replace n with 100 and evaluate the complexity
      c_eval = c.replace('n', '100')
      try:
        # convert it into number first and then compare
        complexity_value = eval(c_eval)
        values.append(complexity_value)
      except Exception as e:
        print(f"Error evaluating complexity {c}: {e}")
        complexity_value = float('inf')
        values.append(complexity_value)

  # get minimal value from the values and return the index
  min_value = min(values)
  best_index = values.index(min_value)
  return best_index

def get_best_solution(o1, o2, o3):
  import os
  import subprocess
  # i need to test the codes over here for efficiency and correctness and then return the best one
  # so we'll have the code with the time complexity measurement, just need to ensure the code is clean with no ```
  o1, o2, o3 = extract_code(o1), extract_code(o2), extract_code(o3)

  os.makedirs("temp_best_solution", exist_ok=True)
  with open("temp_best_solution/solution1.py", "w") as f:
    f.write(o1)
  with open("temp_best_solution/solution2.py", "w") as f:
    f.write(o2)
  with open("temp_best_solution/solution3.py", "w") as f:
    f.write(o3)

  so1 = subprocess.run(["python", "temp_best_solution/solution1.py"], capture_output=True, text=True)
  so2 = subprocess.run(["python", "temp_best_solution/solution2.py"], capture_output=True, text=True)
  so3 = subprocess.run(["python", "temp_best_solution/solution3.py"], capture_output=True, text=True)

  if so1.stderr:
    print(f"Error in solution 1: {so1.stderr}")
  if so2.stderr:
    print(f"Error in solution 2: {so2.stderr}")
  if so3.stderr:
    print(f"Error in solution 3: {so3.stderr}")

   # now we have the outputs and we can compare the time complexities and return the best one
  x = get_best_time_complexity(so1.stdout, so2.stdout, so3.stdout)
  return x

def generate_best_solution(prompt: str, tm1, tm2, tm3, tok1, tok2, tok3, params1, params2, params3):
  prompt = (
    "[SYSTEM]\n"
    "You generate ONLY raw executable Python code.\n"
    "Your response MUST start with valid Python syntax immediately.\n"
    "NEVER output tokens like [SOLUTION], [ANSWER], [CODE], or any brackets.\n"
    "NO explanations. NO markdown. NO comments. NO extra text.\n"
    "If any non-Python text appears, the output is invalid.\n"
    "\n"
    "[TASK]\n"
    + prompt + "\n"
    "\n"
    "[RULES]\n"
    "- Use the most time-efficient algorithm.\n"
    "- Include all imports.\n"
    "- Produce a fully runnable single Python file.\n"
    "- The LAST line MUST be exactly:\n"
    "import time as _t; _s=_t.time(); <call your function with sample args>; "
    "print(f'Time Complexity: O(<expression in n>)')\n"
    "\n"
    "[OUTPUT CONTRACT]\n"
    "Start directly with Python code. Do not add headers or labels."
  )
  gen_params = {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 1024}
  # long generation → vllm hosted backends
  output1 = _vllm_generate(TEACHER_MODEL_IDS[0], [prompt], gen_params)
  output2 = _vllm_generate(TEACHER_MODEL_IDS[1], [prompt], gen_params)
  output3 = _vllm_generate(TEACHER_MODEL_IDS[2], [prompt], gen_params)

  best_output_index=get_best_solution(output1[0]['text'],output2[0]['text'],output3[0]['text'])
  best_output = [output1[0]['text'], output2[0]['text'], output3[0]['text']][best_output_index]
  return best_output

class Node:
  def __init__(self,token_id,generated_text,parent):
    self.token_id=token_id
    self.generated_text=generated_text
    self.parent=parent
    self.children=[]
    self.visit_count=0
    self.value=0.0

  def add_child(self,child_node):
    self.children.append(child_node)


  def get_ucb_score(self):
    import math
    c = 1.414
    """We cant use the puct formula here becuase we use child.prior there meaning we get log_probs tells you how confident it is in that token and based on this the exploration happens but here we need winning and losing traces so if the tokens are bad the logs are bad and mcts wont explore it that much..so use ucb1"""
    if self.visit_count>0:
      exploitation= self.value/self.visit_count
      exploration=c*(math.sqrt(math.log(self.parent.visit_count)/self.visit_count))
    else :
      return float('inf')
    return exploitation + exploration


def get_next_token_logprobs_hf(model, tokenizer, prompt: str, top_k: int = 20):
    import torch
    import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # logits for the last prompt position -> distribution over next token
    logits = outputs.logits[0, -1, :]           # shape: [vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)

    top_log_probs, top_indices = torch.topk(log_probs, top_k)

    result = []
    for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
        token_text = tokenizer.decode([idx])
        result.append((idx, token_text, lp))

    return result  # already sorted descending by logprob

def get_30_tokens(m1, m2, m3, prompt: str, tokenizer1, tokenizer2, tokenizer3):
    formatted1 = get_next_token_logprobs_hf(m1, tokenizer1, prompt)  # highest logprob first
    formatted2 = get_next_token_logprobs_hf(m2, tokenizer2, prompt)
    formatted3 = get_next_token_logprobs_hf(m3, tokenizer3, prompt)

    top5_1 = formatted1[:5]
    bottom5_1 = formatted1[-5:][::-1]

    top5_2 = formatted2[:5]
    bottom5_2 = formatted2[-5:][::-1]

    top5_3 = formatted3[:5]
    bottom5_3 = formatted3[-5:][::-1]

    teachers_agreemnet = (top5_1[0][0] == top5_2[0][0] == top5_3[0][0])

    return [top5_1 + bottom5_1 + top5_2 + bottom5_2 + top5_3 + bottom5_3, teachers_agreemnet]

def expand_leaf(leaf_node, prompt: str, hf_teacher1, hf_teacher2, hf_teacher3, tokenizer1, tokenizer2, tokenizer3):
  prompt=f"For this prompt {prompt} you have generated these texts untill this {leaf_node.generated_text} now generate the next token with this context"
  generate_tokens,teachers_agreement=get_30_tokens(hf_teacher1,hf_teacher2,hf_teacher3,prompt,tokenizer1,tokenizer2,tokenizer3)
  for token in generate_tokens:
    child_node=Node(token[0],leaf_node.generated_text+token[1],leaf_node)
    leaf_node.add_child(child_node)
  return teachers_agreement

def get_all_leaf_nodes(node):
    if len(node.children) == 0:
        return [node]

    leaves = []
    for child in node.children:
        leaves.extend(get_all_leaf_nodes(child))

    return leaves

def get_top_3_leaves(leaf_nodes):
    def get_average_reward(leaf):
        if leaf.visit_count == 0:
            return -float('inf')
        return leaf.value / leaf.visit_count
    sorted_leaves = sorted(leaf_nodes, key=get_average_reward, reverse=True)
    return sorted_leaves[:3]

def get_process_reward(prompt : str, best_solution :str , partial_solution :str, teacher_model1,teacher_model2,teacher_model3,tokenizer1,tokenizer2,tokenizer3) -> float:
  final_prompt=f"""You are an expert code evaluator. Your job is to grade partial, incomplete code on a scale of -1.0 to 1.0 based on whether it is on the right track to solving the user's prompt.
  User Prompt: {prompt}
  Reference Solution (For context only): > {best_solution}
  Partial Code to Evaluate: > {partial_solution}
  CRITICAL INSTRUCTIONS:
  Use the Reference Solution to understand the core logic required.
  DO NOT penalize the Partial Code simply because it uses a different algorithm, different variable names, or a different valid approach than the Reference Solution.
  Only penalize the Partial Code if it contains logic errors, syntax errors, or is going down a provably incorrect path.
  Output only a float between -1.0 and 1.0.

  Output Format:
  You only ouput the float score without any additional text or explanation.Not even any labels or indentation. Just the number."""

  score_params = {"temperature": 0.1, "top_p": 1.0, "max_new_tokens": 10}
  # short scoring generation → vllm hosted backends
  output1 = _vllm_generate(TEACHER_MODEL_IDS[0], [final_prompt], score_params)
  output2 = _vllm_generate(TEACHER_MODEL_IDS[1], [final_prompt], score_params)
  output3 = _vllm_generate(TEACHER_MODEL_IDS[2], [final_prompt], score_params)

  raw_text1 = output1[0]['text']
  raw_text2 = output2[0]['text']
  raw_text3 = output3[0]['text']

  try:
    score1 = float(raw_text1.strip())
  except ValueError:
    score1 = 0.0
  try:
    score2 = float(raw_text2.strip())
  except ValueError:
    score2 = 0.0
  try:
    score3 = float(raw_text3.strip())
  except ValueError:
    score3 = 0.0

  average_score = (score1 + score2 + score3) / 3.0
  return max(-1.0, min(1.0, average_score))  # Clamp to [-1.0, 1.0]

def get_drift_reward(min_len, avg_len, max_len, lengths):
  """
  Calculates a reward based on the internal consistency (variance) of the batch.

  Formula:
      R = 1 - 2 * (stdev / max_tolerable_stdev)^2

  Where:
      stdev = standard deviation of the input lengths
      max_tolerable_stdev = derived from the range (approx (max-min)/2)

  ---------------------------------------------------------------------------
  THEORETICAL JUSTIFICATION & FOUNDATION
  ---------------------------------------------------------------------------

  1. MINIMIZING VARIATION (The "Why"):
    Since we lack a single target, we evaluate the group's "tightness".
    We treat the batch as a process and measure its stability.
    - Reward 1.0: Zero variance (all lengths are identical).
    - Reward -1.0: Maximum variance (points are split between min and max).

  2. ACADEMIC REFERENCES:

    A. Shewhart Cycle & Control Charts (SPC)
        - Source: Shewhart, W. A. (1931). "Economic Control of Quality of
          Manufactured Product".
        - Concept: Shewhart defined "control" not as hitting a number, but as
          minimizing variance ($\sigma$). A process in a state of statistical
          control has a stable, minimized standard deviation.

    B. Coefficient of Variation (Pearson)
        - Source: Pearson, K. (1896). "Mathematical Contributions to the
          Theory of Evolution".
        - Concept: Using relative dispersion ($\sigma / \mu$) ensures the
          reward scales correctly regardless of whether the lengths are
          tiny (0.01) or huge (1000).

  ---------------------------------------------------------------------------
  """
  import math
  all_lengths = [l1 for l1,l2,l3 in lengths] + [l2 for l1,l2,l3 in lengths] + [l3 for l1,l2,l3 in lengths]
  mean_l = sum(all_lengths) / len(all_lengths)
  var = sum((x - mean_l)**2 for x in all_lengths) / len(all_lengths)
  std_len = math.sqrt(var)

  max_std = (max_len - min_len) / 2.0
  if max_std == 0:
      return 1.0

  normalized_std = std_len / max_std
  reward = 1.0 - 2.0 * (normalized_std ** 2)
  return max(-1.0, min(1.0, reward))

def get_cr(completion_reward, test_passed_reward):
  # if test cases are passed only, i consider the completion reward
  if test_passed_reward == 1.0:
    return completion_reward
  else:
    return 0.0

def get_pr(num_leaves):
  return -1.0 if max(3.0, num_leaves) <= 3.0 else 1.0


def get_stop_timing_reward(yes_steps_list: list, max_steps: int) -> list:
    """
    Rewards balanced stopping timing across the group of rollouts.

    Intuition:
    - Too early yes   → risk of incomplete / wrong code   → penalty
    - Very late yes   → inefficient, shallow exploration  → penalty
    - Stopping around ~40–70% of max_steps → often good

    Formula (per rollout):
        normalized_position = yes_step / max_steps
        # Gaussian peaked around 0.55 (tuneable sweet spot)
        r = exp( -(normalized_position - 0.55)^2 / (2 * 0.15^2) ) * 2 - 1

    Group-normalize afterward (fits GRPO/GDPO style — done in compute_grpo_loss).

    Args:
        yes_steps_list: step index when "yes" was output per rollout,
                        or max_steps + 1 if the rollout never stopped early.
        max_steps:      hard cap on MCTS depth from config.

    Returns:
        List of per-rollout raw rewards in [-1, +1].
    """
    import math
    rewards = []
    for step in yes_steps_list:
        if step > max_steps:
            norm_pos = 1.0
        else:
            norm_pos = step / max_steps
        # Gaussian centered ~55% depth, sigma ~15%
        deviation = (norm_pos - 0.55) ** 2
        r = math.exp(-deviation / (2 * 0.15 ** 2))
        scaled = (r * 2) - 1   # → [-1, +1]
        rewards.append(max(-1.0, min(1.0, scaled)))
    return rewards


def get_premature_stop_penalty(yes_decisions: list, partial_texts_at_yes: list) -> list:
    """
    Penalises rollouts that stopped early on clearly incomplete code.

    Returns -1.0 if the rollout declared "yes" but the partial looks unfinished,
    graded up to +0.5 if the partial looks functional.

    Scoring rubric:
        +0.4  has "def " (function defined)
        +0.4  has "return " or "print(" (produces output)
        +0.2  len(text.strip()) > 80 chars  (long enough to do something)
        offset -0.6  → final range ≈ [-0.6, +0.4], clamped to [-1, +0.5]

    Rollouts that did NOT stop (decided_yes=False) receive 0.0 (neutral).

    Args:
        yes_decisions:       List[bool] — True if the rollout terminated with yes.
        partial_texts_at_yes: List[str] — partial code text at the time of yes.

    Returns:
        List of per-rollout raw rewards in [-1.0, +0.5].
    """
    rewards = []
    for decided_yes, text in zip(yes_decisions, partial_texts_at_yes):
        if not decided_yes:
            rewards.append(0.0)   # no penalty if continued
            continue
        has_def     = "def "     in text
        has_return  = "return "  in text or "print(" in text   # proxy for output
        long_enough = len(text.strip()) > 80                   # tune per domain
        score = 0.0
        if has_def:     score += 0.4
        if has_return:  score += 0.4
        if long_enough: score += 0.2
        r = score - 0.6   # maps [0, 1.0] → [-0.6, +0.4]
        rewards.append(max(-1.0, min(0.5, r)))
    return rewards


def get_confidence_calibration(yes_logprobs: list, decided_yes: list) -> list:
    """
    Rewards the student for being confidently correct about its stop/continue decision.

    For "yes" actions: reward high log-prob, penalise hesitant low log-prob.
    For "no"  actions: small bonus for a confident continue.

    Formula:
        if yes:  r = 2.0 * lp - 1.0   (lp near 0 → +1, lp near -1 → -3 → clamped -1)
        if no:   r = lp * 0.5          (small bonus for confident continue)

    Args:
        yes_logprobs: List[float] — log-probability of the action token(s) that
                      triggered the stop/continue decision (negative scalars).
        decided_yes:  List[bool]  — True if the rollout stopped at this step.

    Returns:
        List of per-rollout raw rewards in [-1.0, +1.0].
    """
    rewards = []
    for lp, is_yes in zip(yes_logprobs, decided_yes):
        if is_yes:
            r = 2.0 * lp - 1.0   # lp usually negative
        else:
            r = lp * 0.5
        rewards.append(max(-1.0, min(1.0, r)))
    return rewards