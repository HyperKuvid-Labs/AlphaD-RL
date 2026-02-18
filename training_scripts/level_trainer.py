# i think the env script was a waste of time, wecan directly call the mcts in here, and just start the training
from transformers import AutoTokenizer, AutoModelForCausalLM
import sglang as sgl
import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from structs.main import mcts

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

  # initialize the teacher models and tokenizers (oracles for MCTS)
  tm1 = sgl.Engine(model_path="Qwen/Qwen3-Coder-30B-A3B-Instruct", mem_fraction_static=0.2, context_length=4096)
  tm2 = sgl.Engine(model_path="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", mem_fraction_static=0.2, context_length=4096)
  tm3 = sgl.Engine(model_path="openai/gpt-oss-20b", mem_fraction_static=0.2, context_length=4096)

  tokenizer1 = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
  tokenizer2 = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
  tokenizer3 = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

  params1 = {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 1, "return_logprob": True, "top_logprobs_num": 20}
  params2 = {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 1, "return_logprob": True, "top_logprobs_num": 20}
  params3 = {"temperature": 0.5, "top_p": 1.0, "max_new_tokens": 1, "return_logprob": True, "top_logprobs_num": 20}

  # level_guesser student model (the one we're training with GRPO)
  student_model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen3-4B",
      torch_dtype=torch.bfloat16,
      device_map="auto",
      trust_remote_code=True  # For Qwen models
  )
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
  tokenizer.pad_token = tokenizer.eos_token  # Critical for padding in GRPO

  # Shared cache for MCTS results (prompt -> (top_3_nodes, num_leaves, cr, test_passed_reward))
  # This ensures MCTS runs ONLY ONCE per unique prompt across ALL reward calls and epochs
  mcts_cache = {}

  def get_mcts_results(prompt, test, entry_point):
      """Helper to run/cached MCTS for a prompt."""
      if prompt not in mcts_cache:
          top_3_nodes, num_leaves, cr, test_passed_reward = mcts(
              prompt, test, entry_point, 10, tm1, tm2, tm3, params1, params2, params3, tokenizer1, tokenizer2, tokenizer3
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

  # GRPO config (tune these; vLLM for fast generation during training)
  config = GRPOConfig(
      num_generations=4,          # 4 completions per prompt (for relative advantages)
      max_prompt_length=512,      # Adjust based on your prompts
      max_completion_length=512,  # For code completions
      learning_rate=2e-5,         # Conservative LR for RL
      num_train_epochs=2,         # Or max_steps=1000 for more control
      per_device_train_batch_size=1,  # Small for GPU mem (MCTS is heavy)
      gradient_accumulation_steps=4,
      # Enable SGLang for generation (much faster than HF)
      use_sglang=True,
      # Generation params
      temperature=0.7,
      top_p=0.95,
      # Logging
      report_to="tensorboard",  # Or "wandb"/"tensorboard"
      logging_steps=10,
  )

  # Initialize the trainer with SEPARATE reward_funcs
  # GRPO will call BOTH on the same batch, sum their rewards (or use reward_weights=[1.0, 1.0] in config)
  trainer = GRPOTrainer(
      model=student_model,                    # The level_guesser we're training
      processing_class=tokenizer,             # Tokenizer for the student
      reward_funcs=[completion_reward_func, prune_reward_func],  # List for separate rewards!
      args=config,
      train_dataset=dataset,
  )

  # Start training! (MCTS runs on-demand via cache in the reward funcs)
  trainer.train()

  # Optional: Save the trained level_guesser
  trainer.save_model("level_guesser_qwen3_4b")
