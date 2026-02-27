import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

# ────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────
model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
save_dir   = "adrl-qwen2.5-coder-4bit"

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

quant_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = compute_dtype,
    bnb_4bit_use_double_quant = True,
)

print(f"Loading {model_name} in 4-bit ...")

# ────────────────────────────────────────────────
# Load quantized model + tokenizer
# ────────────────────────────────────────────────

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = quant_config,
    torch_dtype         = compute_dtype,
    device_map          = "auto",
    trust_remote_code   = True,
    attn_implementation = "sdpa",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code = True,
)

print("Model loaded successfully in 4-bit.")

# ────────────────────────────────────────────────
# Save to local folder (safetensors recommended)
# ────────────────────────────────────────────────

print(f"Saving quantized model to: {save_dir}")

os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(
    save_dir,
    safe_serialization = True,          # = safetensors
    # max_shard_size     = "10GB",
)

tokenizer.save_pretrained(save_dir)

print("Done! You can now load it like this:")
print(f'    model = AutoModelForCausalLM.from_pretrained("{save_dir}", device_map="auto")')
