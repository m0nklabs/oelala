import requests
import time
import os

BASE_URL = "http://192.168.1.2:7998"

def test_health():
    print("Testing /health...")
    try:
        res = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
        return res.status_code == 200
    except Exception as e:
        print(f"Failed: {e}")
        return False

def test_list_videos():
    print("\nTesting /list-videos...")
    try:
        res = requests.get(f"{BASE_URL}/list-videos", timeout=5)
        print(f"Status: {res.status_code}")
        data = res.json()
        print(f"Count: {data.get('count')}")
        return res.status_code == 200
    except Exception as e:
        print(f"Failed: {e}")
        return False

def test_generate_text_validation():
    print("\nTesting /generate-text validation (empty prompt)...")
    try:
        res = requests.post(f"{BASE_URL}/generate-text", data={"prompt": ""}, timeout=5)
        print(f"Status: {res.status_code}")
        # Should fail with 400 or 422
        return res.status_code in [400, 422]
    except Exception as e:
        print(f"Failed: {e}")
        return False

if __name__ == "__main__":
    print(f"Smoke testing API at {BASE_URL}")
    if test_health():
        test_list_videos()
        test_generate_text_validation()
    else:
        print("Health check failed, skipping other tests.")
