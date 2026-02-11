#!/usr/bin/env python3
"""
BIPN 145 Fly Tracker — Setup

Run this script once to install the required Python packages.

Usage:
    python setup.py
"""

import subprocess
import sys

packages = [
    ("numpy", "numpy"),
    ("opencv-python", "cv2"),
    ("matplotlib", "matplotlib"),
]

print("=== BIPN 145 Fly Tracker Setup ===\n")

for package, import_name in packages:
    try:
        __import__(import_name)
        print(f"  ✓ {package} is already installed.")
    except ImportError:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
        print(f"  ✓ {package} installed.")

print("\nSetup complete! You can now run the fly tracker with:")
print("  python BIPN145_flytrack.py")
