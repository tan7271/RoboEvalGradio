#!/usr/bin/env python3
"""
Startup script that runs setup.sh before launching the Gradio app.
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
    from app import demo
    demo.launch()

if __name__ == "__main__":
    main()

