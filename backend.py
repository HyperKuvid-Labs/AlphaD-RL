from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7

SGLANG_INSTANCES = {
    "model_1": "",
    "model_2": "",
    "model_3": ""
}

async def fetch_llm_response(client: httpx.AsyncClient, model_name: str, url: str, prompt: str, max_tokens: int, temperature: float):
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }
    }

    try:
        response = await client.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        result_data = response.json()

        return {
            "model": model_name,
            "status": "success",
            "text": result_data.get("text", "")
        }

    except Exception as e:
        return {
            "model": model_name,
            "status": "error",
            "error_message": str(e)
        }

@app.post("/orchestrate")
async def orchestrate_models(request: PromptRequest):
    async with httpx.AsyncClient() as client:

        tasks = []
        for model_name, url in SGLANG_INSTANCES.items():
            task = fetch_llm_response(
                client=client,
                model_name=model_name,
                url=url,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            tasks.append(task)
        results = await asyncio.gather(*tasks)

    final_response = {}
    for res in results:
        final_response[res["model"]] = {
            "status": res["status"],
            "response": res.get("text", None),
            "error": res.get("error_message", None)
        }

    return final_response

