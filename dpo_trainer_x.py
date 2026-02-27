from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
import torch
from trl import DPOTrainer, DPOConfig

model_name = "Qwen/Qwen3-4B"

model = FastLanguageModel.from_pretrained(model_name)

dataset = "Pradheep1647/AlphaD-RL"

tokenizer = AutoTokenizer.from_pretrained(model_name)

dpo_condfig = DPOConfig(
		model_name_or_path = model_name,
		train_dataset      = dataset,
		tokenizer         = tokenizer,
		max_length        = 512,
		num_train_epochs  = 1,
		ref_model_sync=True,
)

trainer = DPOTrainer(config = dpo_condfig, model=model, ref_mod=model)

trainer.train()

trainer.save_pretrained("./adrl-qwen3-4b")