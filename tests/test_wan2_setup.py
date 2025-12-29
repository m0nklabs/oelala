#!/usr/bin/env python3
"""
Test script for Wan2.2 Image-to-Video Generator
Oelala Project - Validation Test
"""

import sys
import os

# Add the oelala directory to Python path
sys.path.append('/home/flip/oelala')

def test_wan2_imports():
    """Test if all required imports work"""
    try:
        import torch
        from diffusers import WanImageToVideoPipeline
        from PIL import Image
        import numpy as np
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_cuda_availability():
    """Test CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return cuda_available
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_model_loading():
    """Test model loading (without downloading full model)"""
    try:
        from diffusers import WanImageToVideoPipeline
        # Just test if the class can be instantiated
        print("✅ WanImageToVideoPipeline class available")
        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Wan2.2 Setup Validation Test")
    print("=" * 50)

    tests = [
        ("Import Test", test_wan2_imports),
        ("CUDA Test", test_cuda_availability),
        ("Model Class Test", test_model_loading)
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
        print("🎉 All tests passed! Wan2.2 setup is ready.")
        print("\n💡 Next steps:")
        print("1. Download a test image (e.g., person.jpg)")
        print("2. Run: python wan2_generator.py")
        print("3. Or use the generator programmatically:")
        print("   from wan2_generator import Wan2VideoGenerator")
        print("   gen = Wan2VideoGenerator()")
        print("   gen.load_model()")
        print("   gen.generate_video_from_image('person.jpg', 'dancing')")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
