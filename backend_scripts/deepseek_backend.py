from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sglang as sgl
from transformers import AutoTokenizer

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = sgl.Engine(
        model_path="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        context_length=4096,
        trust_remote_code=True,
        model_impl="sglang",
    )
    yield
    if model is not None:
        model.shutdown()  # ensure the model is properly shutdown when the server stops

app = FastAPI(lifespan=lifespan)

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7

@app.post("/resp")
def get_resp(data: PromptRequest):
    prompt, max_tokens, temperature = data.prompt, data.max_tokens, data.temperature

    sampling_params = {"temperature": temperature}

    resp = model.generate(prompt, sampling_params)
    return {"response": resp['text']}

@app.get("/tokenizer")
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    return {"tokenizer": tokenizer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)