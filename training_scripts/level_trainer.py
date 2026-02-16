from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from trl import GRPOTrainer, GRPOConfig
# will import the mcts env over here

# rewards over here are:
# - if tasks passed, completion length => {|min-avg| <= 5 => reward 1, else reward -1} else -1
# - no. of nodes pruned in the particular level -> if max(2, x) <=2 -> reward -1 else reward 1


