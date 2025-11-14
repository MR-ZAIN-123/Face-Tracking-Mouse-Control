"""
check_imports.py

Simple helper to verify required packages can be imported in the current
Python environment. Exits with non-zero code if any import fails.
"""
import importlib
import sys

modules = ["cv2", "numpy", "pyautogui"]
all_ok = True

for m in modules:
    try:
        importlib.import_module(m)
        print(f"{m} OK")
    except Exception as e:
        print(f"{m} ERROR: {e}")
        all_ok = False

if not all_ok:
    print("One or more imports failed. See messages above.")
    sys.exit(1)

print("All imports OK")
