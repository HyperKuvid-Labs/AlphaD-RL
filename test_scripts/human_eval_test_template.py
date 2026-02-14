import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def evaulate_model_on_humaneval(model_name):
  # model id, for now: Qwen/Qwen3-4B
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name).to(device)