from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7

@app.on_event("startup")
def startup_event():
    global model
    model = LLM(model="mistralai/Codestral-22B-v0.1", trust_remote_code=True, max_num_seqs=1, max_model_len=4096, tensor_parallel_size=1)


@app.on_event("shutdown")
def shutdown_event():
    global model
    if model is not None:
        model.shutdown()  # ensure the model is properly shutdown when the server stops

@app.post("/resp")
def get_resp(data: PromptRequest):
    prompt, max_tokens, temperature = data.prompt, data.max_tokens, data.temperature

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    resp = model.generate(prompt, sampling_params)
    return {"response": resp[0].outputs[0].text}

@app.get("/tokenizer")
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Codestral-22B-v0.1")
    return {"tokenizer": tokenizer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)