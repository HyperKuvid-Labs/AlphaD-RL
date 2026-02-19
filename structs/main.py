import sglang as sgl
import math
import json
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
import concurrent.futures
import tempfile
import os
import re
import subprocess

class MCTSUpdateCallback(TrainerCallback):
  def __init__(self, student_model, mcts_cache):
    self.student_model = student_model
    self.mcts_cache = mcts_cache

  def on_save(self, args, state, control, **kwargs):
    checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"
    print(f"\n[MCTS Callback] Loading latest checkpoint for MCTS: {checkpoint_path}")

    new_weights = AutoModelForCausalLM.from_pretrained(checkpoint_path).state_dict()
    self.student_model.load_state_dict(new_weights)

    self.mcts_cache.clear()
    print("[MCTS Callback] Updated student model weights and cleared MCTS cache.")

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

c = 1.414

# def execute_and_score(script_content, timeout=5):
#     """Executes code and returns the ratio of passed tests."""
#     # Use delete=False for compatibility with some Windows/Linux edge cases, manual remove later
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
#         tmp.write(script_content)
#         tmp_path = tmp.name

#     try:
#         result = subprocess.run(
#             ["python3", tmp_path],
#             capture_output=True, text=True, timeout=timeout
#         )
#         # Extract "Passed X out of Y"
#         match = re.search(r"Passed\s+(\d+)\s+out\s+of\s+(\d+)\s+test\s+cases", result.stdout)
#         if match:
#             passed, total = int(match.group(1)), int(match.group(2))
#             return passed / total if total > 0 else 0.0
#         return 0.0
#     except (subprocess.TimeoutExpired, Exception):
#         return -1.0 # Penalty for hang or crash
#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)
# def parallel_test_solutions(scripts):
#     """Runs multiple scripts in parallel across CPU cores."""
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         return list(executor.map(execute_and_score, scripts))

def get_best_solution(o1, o2, o3):
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

  x = get_best_time_complexity(so1.stdout, so2.stdout, so3.stdout)
  return x


def calculate_metrics_using_subprocess(script_text: str):
  pass

def level_guesser(model, tokenizer, seq_length, agreement, avg_value, node_count):
    prompt = f"Length:{seq_length}, Agree:{agreement}, Value:{avg_value:.2f}, Nodes:{node_count}. Stop? (Yes/No):"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
      out = model.generate(**inputs, temperature=0.0, top_p=1.0, use_cache=True)

    response = tokenizer.decode(out[0], skip_special_tokens=True).lower()
    return "yes" in response

def generate_best_solution(prompt: str, tm1, tm2, tm3, params1, params2, params3):
  prompt+="Generate the most efficient possible solution for this problem strictly in Python only, output absolutely nothing but the pure Python code itself without any explanations, comments, markdown, text, or extra words whatsoever—ensure the code is correct, optimal, and complete from all necessary imports to the main function or script, including code to measure and print the time complexity at the end of execution."
  params1 = {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 1024}
  params2 = {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 1024}
  params3 = {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 1024}
  output1 = tm1.generate([prompt], params1)
  output2 = tm2.generate([prompt], params2)
  output3 = tm3.generate([prompt], params3)

  best_output_index=get_best_solution(output1[0]['text'],output2[0]['text'],output3[0]['text'])
  best_output = [output1[0]['text'], output2[0]['text'], output3[0]['text']][best_output_index]
  return best_output

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

  params1 = {"temperature": 0.1, "top_p": 1.0, "max_new_tokens": 10}
  params2 = {"temperature": 0.1, "top_p": 1.0, "max_new_tokens": 10}
  params3 = {"temperature": 0.1, "top_p": 1.0, "max_new_tokens": 10}
  output1 = teacher_model1.generate([final_prompt], params1)
  output2 = teacher_model2.generate([final_prompt], params2)
  output3 = teacher_model3.generate([final_prompt], params3)

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
  return average_score

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
    """We cant use the puct formula here becuase we use child.prior there meaning we get log_probs tells you how confident it is in that token and based on this the exploration happens but here we need winning and losing traces so if the tokens are bad the logs are bad and mcts wont explore it that much..so use ucb1"""
    if self.visit_count>0:
      exploitation= self.value/self.visit_count
      exploration=c*(math.sqrt(math.log(self.parent.visit_count)/self.visit_count))
    else :
      return float('inf')
    return exploitation + exploration


def select_leaf_node(root_node):
  current_node=root_node
  while(len(current_node.children)!=0):
    next_node=None
    max_ucb=- float('inf')
    for i in range(0,len(current_node.children)):
      ucb_score=current_node.children[i].get_ucb_score()
      if ucb_score>max_ucb:
        max_ucb=ucb_score
        next_node=current_node.children[i]
    current_node=next_node
  return current_node


def expand_leaf(leaf_node, prompt: str, hf_teacher1, hf_teacher2, hf_teacher3, tokenizer1, tokenizer2, tokenizer3):
  prompt=f"For this prompt {prompt} you have generated these texts untill this {leaf_node.generated_text} now generate the next token with this context"
  generate_tokens,teachers_agreement=get_30_tokens(hf_teacher1,hf_teacher2,hf_teacher3,prompt,tokenizer1,tokenizer2,tokenizer3)
  for token in generate_tokens:
    child_node=Node(token[0],leaf_node.generated_text+token[1],leaf_node)
    leaf_node.add_child(child_node)
  return teachers_agreement

def backpropagte(node , reward):
  node.visit_count+=1
  node.value+=reward
  if node.parent is not None :
    return backpropagte(node.parent , reward)



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

def get_next_token_logprobs_hf(model, tokenizer, prompt: str, top_k: int = 20):
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
  std_len = math.sqrt(sum((l - avg_len) ** 2 for lens in lengths for l in lens) / (3 * len(lengths)))

  max_std = (max_len - min_len) / 2.0
  if max_std == 0:
      return 1.0

  normalized_std = std_len / max_std
  reward = 1.0 - 2.0 * (normalized_std ** 2)
  return max(-1.0, min(1.0, reward))

def mcts(prompt, test, entrypoint, num_simulations, teacher_model1, teacher_model2, teacher_model3, params1, params2, params3, tokenizer1, tokenizer2, tokenizer3, hf_teacher1, hf_teacher2, hf_teacher3, student_model, student_tokenizer):
  root_node=Node(token_id=None,generated_text="",parent=None)
  golden_solution=generate_best_solution(prompt, teacher_model1,teacher_model2,teacher_model3, params1, params2, params3)
  for _ in range(num_simulations) :
    leaf=select_leaf_node(root_node)
    teachers_agreement=expand_leaf(leaf, prompt, hf_teacher1, hf_teacher2, hf_teacher3, tokenizer1, tokenizer2, tokenizer3)
    reward=get_process_reward(prompt,golden_solution,leaf.generated_text, teacher_model1,teacher_model2,teacher_model3,tokenizer1,tokenizer2,tokenizer3)
    backpropagte(leaf,reward)

    seq_length=len(leaf.generated_text)
    if leaf.parent is not None:
      siblings = leaf.parent.children
      node_count = len(siblings)
      total_value = sum(sibling.value for sibling in siblings)
      avg_value = total_value / node_count if node_count > 0 else 0.0
    else:
      node_count = len(leaf.children)
      avg_value = leaf.value
    should_stop = level_guesser(student_model, student_tokenizer, len(leaf.generated_text), teachers_agreement, avg_value, node_count)
    if should_stop:
      "Level Guesser triggered early stop at simulation"
      break

    all_leaves = get_all_leaf_nodes(root_node)
    num_leaves = len(all_leaves)
    top_3_nodes = get_top_3_leaves(all_leaves)

    continuation_prompt = []
    for node in top_3_nodes:
      formatted_prompt = f"""### Context
      {prompt}

      ### Partial Implementation
      {node.generated_text}

      ### Instructions
      1. **Complete the code** starting exactly from where the partial implementation ends. Do NOT repeat the existing code.
      2. **Integrate the Entrypoint**: Ensure the logic flows into the `{entrypoint}` function.
      3. **Main Function & Testing**: Append a `if __name__ == "__main__":` block.
      4. **Validation**: Use the following test cases: {test}.
      5. **Output Format**: The script must conclude by printing the exact string: "Passed X out of Y test cases".

      ### Completion:"""
      continuation_prompt.append(formatted_prompt)

    outputs1 = teacher_model1.generate(continuation_prompt, params1)
    outputs2 = teacher_model2.generate(continuation_prompt, params2)
    outputs3 = teacher_model3.generate(continuation_prompt, params3)

    lengths = []
    test_passed_reward = 1.0
    for i in range(len(top_3_nodes)):
      script1 = outputs1[i]['text']
      script2 = outputs2[i]['text']
      script3 = outputs3[i]['text']

      import os
      os.makedirs("temp_scripts_mcts", exist_ok=True)
      count_passed = 0
      with open(f"temp_scripts_mcts/script1_node{i}.py", "w") as f:
        f.write(script1)
      with open(f"temp_scripts_mcts/script2_node{i}.py", "w") as f:
        f.write(script2)
      with open(f"temp_scripts_mcts/script3_node{i}.py", "w") as f:
        f.write(script3)

      import subprocess

      o1 = subprocess.run(["python", f"temp_scripts_mcts/script1_node{i}.py"], capture_output=True, text=True).stdout
      o2 = subprocess.run(["python", f"temp_scripts_mcts/script2_node{i}.py"], capture_output=True, text=True).stdout
      o3 = subprocess.run(["python", f"temp_scripts_mcts/script3_node{i}.py"], capture_output=True, text=True).stdout

      # removing the temp files
      subprocess.run(["rm", f"temp_scripts_mcts/script1_node{i}.py"])
      subprocess.run(["rm", f"temp_scripts_mcts/script2_node{i}.py"])
      subprocess.run(["rm", f"temp_scripts_mcts/script3_node{i}.py"])

      for output in [o1, o2, o3]:
        import re
         # extract the number of test cases passed from the output
        match = re.search(r"Passed\s+(\d+)\s+out\s+ of\s+(\d+)\s+test\s+cases", output)
        if match:
          passed = int(match.group(1))
          total = int(match.group(2))
          if passed == total:
            count_passed += 1

      if count_passed < 1 and test_passed_reward != -1.0:
        test_passed_reward = -1.0

      len1 = len(script1)
      len2 = len(script2)
      len3 = len(script3)

      lengths.append((len1, len2, len3))

    min_len = min(min(lens) for lens in lengths)
    avg_len = sum(sum(lens) for lens in lengths) / (3 * len(lengths))
    max_len = max(max(lens) for lens in lengths)

    # reward for the completion length
    cr = get_drift_reward(min_len, avg_len, max_len, lengths)
    return top_3_nodes, num_leaves, cr, test_passed_reward

def calculate_cyclomatic_complexity(code_string: str) -> int:
    complexity_score = 1
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return 100
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.BoolOp)):
            complexity_score += 1

    return complexity_score


# def calculate_readability_score(code_string: str) -> float:
#   prompt = f"""You are an expert Python code reviewer. Evaluate the following code based on readability, clear variable naming, and overall elegance.
#   Score it on a scale from 1 to 10, where 1 is unreadable spaghetti code and 10 is perfectly clean, production-ready code.

#   Code to Evaluate:
#   {code_string}

#   CRITICAL INSTRUCTIONS:
#   Output ONLY a single integer between 1 and 10. Do not output any text, markdown, explanation, or punctuation. Just the number.
#   """

#   params = SamplingParams(
#       temperature=0.1,
#       top_p=1.0,
#       max_tokens=5
#   )
#   output1 = teacher_model1.generate(prompts=[prompt], sampling_params=params)
#   output2 = teacher_model2.generate(prompts=[prompt], sampling_params=params)
#   output3 = teacher_model3.generate(prompts=[prompt], sampling_params=params)

#   raw_text1 = output1[0].outputs[0].text
#   raw_text2 = output2[0].outputs[0].text
#   raw_text3 = output3[0].outputs[0].text

#   try:
#     score1 = float(raw_text1.strip())
#   except ValueError:
#     score1 = 0.0
#   try:
#     score2 = float(raw_text2.strip())
#   except ValueError:
#     score2 = 0.0
#   try:
#     score3 = float(raw_text3.strip())
#   except ValueError:
#     score3 = 0.0

#   average_score = (score1 + score2 + score3) / 3.0
#   return average_score

# def evaluate_code_quality(script_text: str) -> float:
#   exec_time, peak_memory = calculate_metrics_using_subprocess(script_text)
#   cyclomatic_complexity = calculate_cyclomatic_complexity(script_text)
#   readability_score = calculate_readability_score(script_text)
#   time_score = 1.0 / (1.0 + exec_time)
#   mem_score = 1.0 / (1.0 + math.log10(peak_memory + 1.0))
#   comp_score = 1.0 / cyclomatic_complexity
#   read_score = readability_score / 10.0
#   final_quality_score = (0.40 * time_score) + (0.20 * mem_score) + (0.20 * comp_score) + (0.20 * read_score)
#   return final_quality_score


def finish_and_extract_dpo(prompt: str, top_3_nodes: list, llm1, llm2, llm3, dataset_path="dpo_dataset.jsonl"):
    params = {"temperature": 0.2, "top_p": 0.95, "max_new_tokens": 1024}

    continuation_prompts = []
    for node in top_3_nodes:
        formatted_prompt = f"{prompt}\n\nContinue and complete the following code without repeating what is already written:\n{node.generated_text}"
        continuation_prompts.append(formatted_prompt)

    outputs1 = llm1.generate(continuation_prompts, params)
    outputs2 = llm2.generate(continuation_prompts, params)
    outputs3 = llm3.generate(continuation_prompts, params)

    final_3_scripts = []

    # for i in range(len(top_3_nodes)):
    #     partial_code = top_3_nodes[i].generated_text

    #     script1 = partial_code + outputs1[i].outputs[0].text
    #     script2 = partial_code + outputs2[i].outputs[0].text
    #     script3 = partial_code + outputs3[i].outputs[0].text

    #     score1 = evaluate_code_quality(script1)
    #     score2 = evaluate_code_quality(script2)
    #     score3 = evaluate_code_quality(script3)

    #     best_tuple_for_node = max(
    #         [(script1, score1), (script2, score2), (script3, score3)],
    #         key=lambda x: x[1]
    #     )

    #     final_3_scripts.append(best_tuple_for_node)
    scored_finals = sorted(final_3_scripts, key=lambda x: x[1], reverse=True)

    chosen_script = scored_finals[0][0]
    rejected_script = scored_finals[-1][0]
    dpo_pair = {
        "prompt": prompt,
        "chosen": chosen_script,
        "rejected": rejected_script
    }

    with open(dataset_path, "a") as f:
        f.write(json.dumps(dpo_pair) + "\n")


    return dpo_pair

# i think the env script was a waste of time, wecan directly call the mcts in here, and just start the training

def get_cr(completion_reward, test_passed_reward):
  # if test cases are passed only, i consider the completion reward
  if test_passed_reward == 1.0:
    return completion_reward
  else:
    return 0.0

def get_pr(num_leaves):
  return -1.0 if max(3.0, num_leaves) <= 3.0 else 1.0

if __name__ == "__main__":
  # load the dataset (HumanEval uses "test" split with 164 examples)
  dataset = load_dataset("openai/openai_humaneval", split="test")

  # initialize the teacher models (sglang engines) for generation (completions, process reward, etc.)
  tm1 = sgl.Engine(model_path="openai/gpt-oss-20b", mem_fraction_static=0.25, context_length=4096)
  tm2 = sgl.Engine(model_path="Qwen/Qwen2.5-Coder-14B-Instruct", mem_fraction_static=0.25, context_length=4096)
  tm3 = sgl.Engine(model_path="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", mem_fraction_static=0.25, context_length=4096)

  tokenizer1 = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
  tokenizer2 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
  tokenizer3 = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")

  # HF models used exclusively for next-token logprob computation in MCTS expansion
  hf_tm1 = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", device_map="auto", trust_remote_code=True)
  hf_tm2 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct", device_map="auto", trust_remote_code=True)
  hf_tm3 = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", device_map="auto", trust_remote_code=True)

  params1 = {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 1024}
  params2 = {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 1024}
  params3 = {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 1024}

  # level_guesser student model (the one we're training with GRPO)
  # student_model = AutoModelForCausalLM.from_pretrained(
  #     "Qwen/Qwen3-4B",
  #     device_map="auto",
  #     trust_remote_code=True
  # )
  # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
  # tokenizer.pad_token = tokenizer.eos_token  # Critical for padding in GRPO
  # student_model, tokenizer = FastLanguageModel.from_pretrained(
  #   model_name="Qwen/Qwen3-4B",
  #   gpu_memory_utilization=0.25,
  #   load_in_4bit=False,
  #   max_seq_len=4096,
  #   dtype="auto"
  # )
  student_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", device_map="auto", trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
  tokenizer.pad_token = tokenizer.eos_token

  # Shared cache for MCTS results (prompt -> (top_3_nodes, num_leaves, cr, test_passed_reward))
  # This ensures MCTS runs ONLY ONCE per unique prompt across ALL reward calls and epochs
  mcts_cache = {}

  mcts_callback = MCTSUpdateCallback(student_model, mcts_cache)

  def get_mcts_results(prompt, test, entry_point):
      if prompt not in mcts_cache:
          top_3_nodes, num_leaves, cr, test_passed_reward = mcts(
              prompt, test, entry_point, 10, tm1, tm2, tm3, params1, params2, params3, tokenizer1, tokenizer2, tokenizer3, hf_tm1, hf_tm2, hf_tm3, mcts_callback.student_model, tokenizer
          )
          mcts_cache[prompt] = (top_3_nodes, num_leaves, cr, test_passed_reward)
      return mcts_cache[prompt]

  # SEPARATE reward function #1: Completion reward (from get_cr)
  # Called by GRPO for every batch of generations; uses MCTS cache
  def completion_reward_func(prompts, completions, test, entry_point, **kwargs):
      rewards = []
      for prompt, tests, ep in zip(prompts, test, entry_point):
          _, num_leaves, cr, test_passed_reward = get_mcts_results(prompt, tests, ep)
          completion_reward = get_cr(cr, test_passed_reward)
          rewards.append(completion_reward)
      return rewards

  # SEPARATE reward function #2: Prune reward (from get_pr)
  # Called by GRPO for every batch of generations; uses MCTS cache
  def prune_reward_func(prompts, completions, test, entry_point, **kwargs):
      rewards = []
      for prompt, tests, ep in zip(prompts, test, entry_point):
          _, num_leaves, _, _ = get_mcts_results(prompt, tests, ep)
          prune_error = get_pr(num_leaves)
          rewards.append(prune_error)
      return rewards

  config = GRPOConfig(
      num_generations=4,          # 4 completions per prompt (for relative advantages)
      max_prompt_length=512,      # Adjust based on your prompts
      max_completion_length=512,  # For code completions
      learning_rate=2e-5,         # Conservative LR for RL
      num_train_epochs=2,         # Or max_steps=1000 for more control
      per_device_train_batch_size=1,  # Small for GPU mem (MCTS is heavy)
      gradient_accumulation_steps=4,
      # Enable SGLang for generation (much faster than HF)
      shuffle_dataset=True,
      # Generation params
      temperature=0.7,
      top_p=0.95,
      # Logging
      report_to="tensorboard",  # Or "wandb"/"tensorboard"
      logging_steps=10,
      output_dir="./grpo_output",  # Where checkpoints and logs go
      loss_type="dapo",
      epsilon=0.3, # clip range thing, 1+ε, 1-ε
      beta=1.0, # kl coeff
      delta=1.3, # double clipping, idea from INTELLECT-2

      log_completions=True, # log a pair of (prompt, completion) every logging_steps

      # multi_objective_aggregation="normalize_then_sum" # gdpo style
      # cannot use this, as it is available only in trl>=0.27.0 but for that we need vlm<0.13.0, and if i pin vllm==0.12.0, it requires torch<=2.9.0, but im currently using torch >=2.9.1, so thought og not changing the whole cycle
  )

  trainer = GRPOTrainer(
      model=student_model,                    # The level_guesser we're training
      processing_class=tokenizer,             # Tokenizer for the student
      reward_funcs=[completion_reward_func, prune_reward_func],  # List for separate rewards!
      args=config,
      train_dataset=dataset,
      callbacks=[mcts_callback]
  )

  # Start training! (MCTS runs on-demand via cache in the reward funcs)
  trainer.train(resume_from_checkpoint=True)

  # Optional: Save the trained level_guesser
  trainer.save_model("level_guesser_qwen3_4b")
