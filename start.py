#!/usr/bin/env python3
"""
LOCAL DEVELOPMENT STARTUP SCRIPT

This script is for LOCAL DEVELOPMENT ONLY.

For HuggingFace Spaces deployment:
  - Docker Spaces use Dockerfile and run.sh
  - This script is NOT used in Docker Spaces
  - Use this script only when testing locally outside Docker

Usage (local development):
  python start.py
"""
import subprocess
import sys
import os

def main():
    """Run setup script then launch app."""
    print("Running dependency setup...")
    
    # Run setup script
    result = subprocess.run(["bash", "setup.sh"], cwd=os.path.dirname(__file__))
    
    if result.returncode != 0:
        print("Setup failed!")
        sys.exit(1)
    
    print("\nLaunching Gradio app...")
    
    # Import and run the app
    from app import create_gradio_interface
    demo = create_gradio_interface()
    demo.launch()

if __name__ == "__main__":
    main()

