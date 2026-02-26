import sys
import requests
from utils import TEACHER_ENDPOINTS

def test_backends() -> bool:
    all_ok = True
    for model_id, url in TEACHER_ENDPOINTS.items():
        try:
            r = requests.post(f"{url}/resp", json={"prompt": "hi", "max_tokens": 4}, timeout=10)
            r.raise_for_status()
            print(f"[OK]   {model_id}  ({url})")
        except Exception as e:
            print(f"[FAIL] {model_id}  ({url}) — {e}")
            all_ok = False
    return all_ok

if __name__ == "__main__":
    if not test_backends():
        print("One or more backends unreachable. Exiting.")
        sys.exit(1)
    print("All backends OK.")
