#!/usr/bin/env python3
"""
Quick smoke test for REST API v1.
Tests basic endpoints without requiring authentication.
"""

import requests
import sys
import os

# Default to localhost, but allow override via environment variable
BASE_URL = os.getenv("OELALA_API_URL", "http://localhost:7998")


def test_api_v1_health():
    """Test API v1 health endpoint."""
    print("Testing GET /api/v1/health...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/health", timeout=5)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Response: {data}")
            assert data["status"] == "healthy"
            assert "version" in data
            print("  ✅ PASS")
            return True
        else:
            print(f"  ❌ FAIL: Expected 200, got {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_api_v1_generate_no_auth():
    """Test that generation requires authentication."""
    print("\nTesting POST /api/v1/generate (no auth)...")
    try:
        resp = requests.post(
            f"{BASE_URL}/api/v1/generate",
            json={"type": "text-to-image", "prompt": "test"},
            timeout=5,
        )
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 401:
            print(f"  Response: {resp.json()}")
            print("  ✅ PASS (correctly requires auth)")
            return True
        else:
            print(f"  ❌ FAIL: Expected 401, got {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_api_v1_credits_no_auth():
    """Test that credits endpoint requires authentication."""
    print("\nTesting GET /api/v1/credits (no auth)...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/credits", timeout=5)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 401:
            print(f"  Response: {resp.json()}")
            print("  ✅ PASS (correctly requires auth)")
            return True
        else:
            print(f"  ❌ FAIL: Expected 401, got {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_api_v1_invalid_key():
    """Test with invalid API key."""
    print("\nTesting with invalid API key...")
    try:
        resp = requests.post(
            f"{BASE_URL}/api/v1/generate",
            headers={"X-API-Key": "oelala_invalid_key_12345"},
            json={"type": "text-to-image", "prompt": "test"},
            timeout=5,
        )
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 401:
            print(f"  Response: {resp.json()}")
            print("  ✅ PASS (invalid key rejected)")
            return True
        else:
            print(f"  ❌ FAIL: Expected 401, got {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_api_keys_management_no_jwt():
    """Test that API key management requires JWT."""
    print("\nTesting GET /api/keys (no JWT)...")
    try:
        resp = requests.get(f"{BASE_URL}/api/keys", timeout=5)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 403:  # FastAPI returns 403 for missing bearer token
            print(f"  Response: {resp.json()}")
            print("  ✅ PASS (correctly requires JWT)")
            return True
        else:
            print(f"  ❌ FAIL: Expected 403, got {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


if __name__ == "__main__":
    print(f"🔬 Smoke testing REST API v1 at {BASE_URL}\n")
    print("=" * 60)

    results = []

    # Run tests
    results.append(test_api_v1_health())
    results.append(test_api_v1_generate_no_auth())
    results.append(test_api_v1_credits_no_auth())
    results.append(test_api_v1_invalid_key())
    results.append(test_api_keys_management_no_jwt())

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All smoke tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
