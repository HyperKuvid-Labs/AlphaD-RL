from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
import torch
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

model_name = "Qwen/Qwen3-4B"

dataset = load_dataset("Pradheep1647/adrl-dpo")

model = FastLanguageModel.from_pretrained(
  model_name,
  max_seq_length=4096,
  load_in_4bit=False,
  full_finetuning=True,
  trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

dpo_config = DPOConfig(
		do_train=True,
    per_device_train_batch_size=1,
    torch_empty_cache_steps=8,
    num_train_epochs=3,
    max_steps=500,
    learning_rate=1e-5,
    logging_steps=10,
    sync_ref_model=True,
    ref_model_sync_steps=32
)

trainer = DPOTrainer(
		model=model,
    ref_model=model,
    train_dataset=dataset["train"],
    args=dpo_config,
		tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("adrl-qwen3-4b")