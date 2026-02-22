from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sglang as sgl
from transformers import AutoTokenizer

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7

@app.on_event("startup")
def startup_event():
    global model
    model = sgl.Engine(model_path="mistralai/Codestral-22B-v0.1", context_length=4096, trust_remote_code=True, mem_fraction_static=0.8, disable_cuda_graph=True)


@app.on_event("shutdown")
def shutdown_event():
    global model
    if model is not None:
        model.shutdown()  # ensure the model is properly shutdown when the server stops

@app.post("/resp")
def get_resp(data: PromptRequest):
    prompt, max_tokens, temperature = data.prompt, data.max_tokens, data.temperature

    sampling_params = {"temperature": temperature}

    resp = model.generate(prompt, sampling_params)
    return {"response": resp['text']}

@app.get("/tokenizer")
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Codestral-22B-v0.1")
    return {"tokenizer": tokenizer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)