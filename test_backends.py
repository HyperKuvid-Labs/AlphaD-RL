import sys
import requests
from utils import TEACHER_ENDPOINTS
import time

def test_backends() -> bool:
    all_ok = True
    url = "http://100.83.38.38:8000"
    model_id = "gpt-oss-20b"
    try:
        start = time.time()
        r = requests.post(f"{url}/v1/completions", json={
    "prompt": "### Instruction:\nWrite C++ code to add two numbers.No explanation, just code.\n### Response:\n",
    "max_tokens": 2048
})
        end = time.time()
        total_time = end - start
        print(f"Response time for {model_id}: {total_time:.2f} seconds")
        r.raise_for_status()
        output = r.json()
        text = output.get("choices", [{}])[0].get("text", "")
        print(f"[OK]   {model_id}  ({url}) — Response: {text}")
    except Exception as e:
        print(f"[FAIL] {model_id}  ({url}) — {e}")
        all_ok = False
    return all_ok

if __name__ == "__main__":
    if not test_backends():
        print("One or more backends unreachable. Exiting.")
        sys.exit(1)
    print("All backends OK.")
