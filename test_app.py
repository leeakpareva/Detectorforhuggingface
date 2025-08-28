"""
NAVADA 2.0 End-to-End Test Suite
Comprehensive testing for all app features
"""

import sys
import os
import time
import json
import numpy as np
from PIL import Image
from datetime import datetime

# Test results storage
test_results = {
    "timestamp": datetime.now().isoformat(),
    "tests_run": 0,
    "tests_passed": 0,
    "tests_failed": 0,
    "failures": [],
    "performance": {},
    "components": {}
}

def test_imports():
    """Test all required imports"""
    print("\n=== Testing Imports ===")
    imports_ok = True
    required_imports = [
        ("streamlit", "Streamlit framework"),
        ("torch", "PyTorch for AI models"),
        ("ultralytics", "YOLO object detection"),
        ("openai", "OpenAI API client"),
        ("cv2", "OpenCV for image processing"),
        ("plotly", "Plotly for charts"),
        ("pandas", "Data processing"),
        ("numpy", "Numerical operations"),
        ("PIL", "Image handling")
    ]
    
    for module_name, description in required_imports:
        try:
            __import__(module_name)
            print(f"[OK] {description} ({module_name})")
            test_results["components"][module_name] = "OK"
        except ImportError as e:
            print(f"[FAIL] {description} ({module_name}): {e}")
            test_results["components"][module_name] = "FAILED"
            test_results["failures"].append(f"Import {module_name}: {e}")
            imports_ok = False
    
    return imports_ok

def test_backend_modules():
    """Test backend module loading"""
    print("\n=== Testing Backend Modules ===")
    backend_ok = True
    
    modules = [
        "backend.yolo",
        "backend.openai_client",
        "backend.face_detection",
        "backend.recognition",
        "backend.database"
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"[OK] {module}")
            test_results["components"][module] = "OK"
        except Exception as e:
            print(f"[FAIL] {module}: {e}")
            test_results["components"][module] = f"FAILED: {e}"
            test_results["failures"].append(f"Backend {module}: {e}")
            backend_ok = False
    
    return backend_ok

def test_yolo_detection():
    """Test YOLO object detection"""
    print("\n=== Testing YOLO Detection ===")
    try:
        from backend.yolo import detect_objects
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = [255, 255, 255]  # White image
        
        start_time = time.time()
        detected_img, detected_objects = detect_objects(test_image)
        detection_time = time.time() - start_time
        
        test_results["performance"]["yolo_detection"] = detection_time
        
        print(f"[OK] YOLO detection working")
        print(f"  Detection time: {detection_time:.3f}s")
        print(f"  Objects detected: {len(detected_objects)}")
        
        test_results["components"]["yolo_detection"] = "OK"
        return True
    except Exception as e:
        print(f"[FAIL] YOLO detection failed: {e}")
        test_results["components"]["yolo_detection"] = f"FAILED: {e}"
        test_results["failures"].append(f"YOLO: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection"""
    print("\n=== Testing OpenAI Connection ===")
    try:
        from backend.openai_client import explain_detection
        
        test_objects = ["person", "laptop", "cup"]
        start_time = time.time()
        explanation = explain_detection(test_objects)
        api_time = time.time() - start_time
        
        test_results["performance"]["openai_api"] = api_time
        
        if explanation and len(explanation) > 0:
            print(f"[OK] OpenAI API working")
            print(f"  Response time: {api_time:.3f}s")
            print(f"  Response length: {len(explanation)} chars")
            test_results["components"]["openai_api"] = "OK"
            return True
        else:
            raise Exception("Empty response from OpenAI")
    except Exception as e:
        print(f"[FAIL] OpenAI API failed: {e}")
        test_results["components"]["openai_api"] = f"FAILED: {e}"
        test_results["failures"].append(f"OpenAI: {e}")
        return False

def test_database():
    """Test database operations"""
    print("\n=== Testing Database ===")
    try:
        from backend.database import db
        
        # Test getting stats
        stats = db.get_stats()
        
        print(f"[OK] Database working")
        print(f"  Faces in DB: {stats.get('faces', 0)}")
        print(f"  Objects in DB: {stats.get('objects', 0)}")
        print(f"  Total detections: {stats.get('total_detections', 0)}")
        
        test_results["components"]["database"] = "OK"
        return True
    except Exception as e:
        print(f"[FAIL] Database failed: {e}")
        test_results["components"]["database"] = f"FAILED: {e}"
        test_results["failures"].append(f"Database: {e}")
        return False

def test_face_detection():
    """Test face detection module"""
    print("\n=== Testing Face Detection ===")
    try:
        from backend.face_detection import face_detector
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = [255, 255, 255]
        
        start_time = time.time()
        detected_img, face_stats = face_detector.detect_faces(test_image)
        face_time = time.time() - start_time
        
        test_results["performance"]["face_detection"] = face_time
        
        print(f"[OK] Face detection working")
        print(f"  Detection time: {face_time:.3f}s")
        print(f"  Faces found: {face_stats.get('total_faces', 0)}")
        
        test_results["components"]["face_detection"] = "OK"
        return True
    except Exception as e:
        print(f"[FAIL] Face detection failed: {e}")
        test_results["components"]["face_detection"] = f"FAILED: {e}"
        test_results["failures"].append(f"Face Detection: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("\n=== Testing Environment ===")
    env_ok = True
    
    # Check for required environment variables
    env_vars = ["OPENAI_API_KEY"]
    
    for var in env_vars:
        if os.getenv(var):
            print(f"[OK] {var} is set")
            test_results["components"][f"env_{var}"] = "OK"
        else:
            print(f"[FAIL] {var} is not set")
            test_results["components"][f"env_{var}"] = "MISSING"
            test_results["failures"].append(f"Environment variable {var} not set")
            env_ok = False
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  Python version: {python_version}")
    test_results["components"]["python_version"] = python_version
    
    return env_ok

def test_gpu_support():
    """Test GPU/CUDA support"""
    print("\n=== Testing GPU Support ===")
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[OK] CUDA GPU available: {gpu_name}")
            print(f"  GPU Memory: {gpu_memory:.2f} GB")
            test_results["components"]["gpu"] = f"OK - {gpu_name}"
        else:
            print("[FAIL] No CUDA GPU available (CPU mode)")
            test_results["components"]["gpu"] = "CPU only"
        
        return True
    except Exception as e:
        print(f"[FAIL] GPU check failed: {e}")
        test_results["components"]["gpu"] = f"FAILED: {e}"
        return False

def generate_report():
    """Generate test report"""
    print("\n" + "="*60)
    print("NAVADA 2.0 - END-TO-END TEST REPORT")
    print("="*60)
    
    # Summary
    total = test_results["tests_run"]
    passed = test_results["tests_passed"]
    failed = test_results["tests_failed"]
    
    print(f"\nTest Summary:")
    print(f"  Total Tests: {total}")
    print(f"  Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"  Failed: {failed} ({failed/total*100:.1f}%)")
    
    # Performance metrics
    if test_results["performance"]:
        print(f"\nPerformance Metrics:")
        for metric, value in test_results["performance"].items():
            print(f"  {metric}: {value:.3f}s")
    
    # Component status
    print(f"\nComponent Status:")
    for component, status in test_results["components"].items():
        if "OK" in str(status):
            print(f"  [OK] {component}: {status}")
        else:
            print(f"  [FAIL] {component}: {status}")
    
    # Failures
    if test_results["failures"]:
        print(f"\nFailures ({len(test_results['failures'])}):")
        for failure in test_results["failures"]:
            print(f"  • {failure}")
    
    # Overall status
    print(f"\n{'='*60}")
    if failed == 0:
        print("[OK] ALL TESTS PASSED - App is fully functional!")
    elif failed < total * 0.3:
        print("⚠ PARTIAL SUCCESS - Most features working")
    else:
        print("[FAIL] CRITICAL ISSUES - Multiple failures detected")
    print("="*60)
    
    # Save report to file
    with open("test_report.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nDetailed report saved to test_report.json")

def main():
    """Run all tests"""
    print("Starting NAVADA 2.0 End-to-End Testing...")
    print("Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    tests = [
        ("Imports", test_imports),
        ("Backend Modules", test_backend_modules),
        ("Environment", test_environment),
        ("GPU Support", test_gpu_support),
        ("Database", test_database),
        ("YOLO Detection", test_yolo_detection),
        ("Face Detection", test_face_detection),
        ("OpenAI Connection", test_openai_connection)
    ]
    
    for test_name, test_func in tests:
        test_results["tests_run"] += 1
        try:
            if test_func():
                test_results["tests_passed"] += 1
            else:
                test_results["tests_failed"] += 1
        except Exception as e:
            print(f"[FAIL] {test_name} crashed: {e}")
            test_results["tests_failed"] += 1
            test_results["failures"].append(f"{test_name} crash: {e}")
    
    generate_report()

if __name__ == "__main__":
    main()