# this will be the env script for the level guesser training, so it just contains the mcts implementation with populating the nodes, pruning the nodes, and just some other funcs like extracting the rewards like (test_cases passed, completion length, no. of nodes pruned) and the level guesser function itself which will be a simple function that takes in the above mentioned rewards and gives out the level to prune at

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import math
from structs.main import mcts
from datasets import load_dataset
import os
import subprocess

# for ref:
# def mcts(prompt , num_simulations) :
#   root_node=Node(token_id=None,generated_text="",parent=None)
#   golden_solution=generate_best_solution(prompt)
#   for _ in range(num_simulations) :
#     leaf=select_leaf_node(root_node)
#     teachers_agreement=expand_leaf(leaf,prompt)
#     reward=get_process_reward(prompt,golden_solution,leaf.generated_text)
#     backpropagte(leaf,reward)

#     seq_length=len(leaf.generated_text)
#     if leaf.parent is not None:
#       siblings = leaf.parent.children
#       node_count = len(siblings)
#       total_value = sum(sibling.value for sibling in siblings)
#       avg_value = total_value / node_count if node_count > 0 else 0.0
#     else:
#       node_count = len(leaf.children)
#       avg_value = leaf.value

#     should_stop = level_guesser(seq_length, teachers_agreement, avg_value, node_count)
#     if should_stop:
#       all_leaves = get_all_leaf_nodes(root_node)
#       top_3_nodes = get_top_3_leaves(all_leaves)
#       return top_3_nodes

class LevelGuesserEnv:
  def __init__(self, tm1, tm2, tm3):
    self.tm1 = LLM(tm1)
    self.tm2 = LLM(tm2)
    self.tm3 = LLM(tm3)

    self.tokenizer1 = self.tm1.get_tokenizer()
    self.tokenizer2 = self.tm2.get_tokenizer()
    self.tokenizer3 = self.tm3.get_tokenizer()

    vocab_size1 = self.tokenizer1.vocab_size
    vocab_size2 = self.tokenizer2.vocab_size
    vocab_size3 = self.tokenizer3.vocab_size

    self.params1 = SamplingParams(
        temperature=0.5,
        top_p=1.0,
        max_tokens=1,
        logprobs=vocab_size1
    )
    self.params2 = SamplingParams(
        temperature=0.5,
        top_p=1.0,
        max_tokens=1,
        logprobs=vocab_size2
    )
    self.params3 = SamplingParams(
        temperature=0.5,
        top_p=1.0,
        max_tokens=1,
        logprobs=vocab_size3
    )

    self.dataset = load_dataset("openai/openai_humaneval", split="test")

    


