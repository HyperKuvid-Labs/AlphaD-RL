from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sglang as sgl
from transformers import AutoTokenizer

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7

@app.post("/resp")
def get_resp(data: PromptRequest):
    prompt, max_tokens, temperature = data.prompt, data.max_tokens, data.temperature

    model = sgl.Engine("Qwen/Qwen2.5-Coder-14B-Instruct", context_length=4096)
    sampling_params = {"temperature": temperature}

    resp = model.generate(prompt, sampling_params)
    model.shutdown()  # the model is shutdown after evry request, this is unoptimal for now, but would suffice my use case
    return {"response": resp['text']}

@app.get("/tokenizer")
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
    return {"tokenizer": tokenizer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)