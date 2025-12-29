#!/usr/bin/env python3
"""
Oelala Web Interface Test Script
Tests the backend and frontend components
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def test_backend_imports():
    """Test if backend dependencies can be imported"""
    print("🔍 Testing backend imports...")

    try:
        sys.path.append('/home/flip/oelala')
        from wan2_generator import Wan2VideoGenerator
        print("✅ Wan2VideoGenerator import successful")

        # Test FastAPI imports
        import fastapi
        import uvicorn
        print("✅ FastAPI imports successful")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_frontend_setup():
    """Test if frontend dependencies are installed"""
    print("🔍 Testing frontend setup...")

    frontend_dir = Path("/home/flip/oelala/src/frontend")

    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False

    if not (frontend_dir / "node_modules").exists():
        print("❌ Frontend dependencies not installed")
        print("💡 Run: cd /home/flip/oelala/src/frontend && npm install")
        return False

    if not (frontend_dir / "package.json").exists():
        print("❌ package.json not found")
        return False

    print("✅ Frontend setup looks good")
    return True

def test_directories():
    """Test if required directories exist"""
    print("🔍 Testing directory structure...")

    dirs_to_check = [
        "/home/flip/oelala/uploads",
        "/home/flip/oelala/generated",
        "/home/flip/oelala/src/backend",
        "/home/flip/oelala/src/frontend"
    ]

    all_exist = True
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
            all_exist = False

    return all_exist

def test_startup_script():
    """Test if startup script exists and is executable"""
    print("🔍 Testing startup script...")

    script_path = "/home/flip/oelala/start_web.sh"

    if not os.path.exists(script_path):
        print("❌ Startup script not found")
        return False

    if not os.access(script_path, os.X_OK):
        print("❌ Startup script not executable")
        print("💡 Run: chmod +x /home/flip/oelala/start_web.sh")
        return False

    print("✅ Startup script ready")
    return True

def main():
    """Run all tests"""
    print("🧪 Oelala Web Interface Test Suite")
    print("=" * 50)

    tests = [
        ("Backend Imports", test_backend_imports),
        ("Frontend Setup", test_frontend_setup),
        ("Directory Structure", test_directories),
        ("Startup Script", test_startup_script)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Web interface is ready.")
        print("\n🚀 To start the web interface:")
        print("   cd /home/flip/oelala")
        print("   ./scripts/start_web.sh")
        print("\n🌐 Then open:")
        print("   Frontend: http://192.168.1.2:5174")
        print("   Backend:  http://192.168.1.2:7998")
    else:
        print("⚠️ Some tests failed. Please fix the issues above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
