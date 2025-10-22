#!/usr/bin/env python3
"""
Upload pi0_CubeHandover checkpoint to Hugging Face Hub
"""

from huggingface_hub import HfApi, create_repo
import os

# Configuration
CHECKPOINT_DIR = "pi0_CubeHandover_ckpt"
REPO_ID = "tan7271/pi0_CubeHandover_ckpt"  # Change "tan7271" to your HF username if different
REPO_TYPE = "model"

def upload_checkpoint():
    """Upload checkpoint to Hugging Face Hub"""
    
    # Check if checkpoint directory exists
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Error: Checkpoint directory '{CHECKPOINT_DIR}' not found!")
        return
    
    print(f"📦 Uploading checkpoint from: {CHECKPOINT_DIR}")
    print(f"🎯 Target repository: {REPO_ID}")
    print()
    
    # Initialize Hugging Face API
    api = HfApi()
    
    try:
        # Create repository (if it doesn't exist)
        print("Creating repository on Hugging Face Hub...")
        create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            exist_ok=True,  # Don't fail if repo already exists
            private=False,  # Set to True if you want a private repo
        )
        print(f"✅ Repository created/verified: https://huggingface.co/{REPO_ID}")
        print()
        
        # Upload the checkpoint folder (using upload_large_folder for large checkpoints)
        print("📤 Uploading checkpoint files...")
        print("(This may take a while depending on checkpoint size)")
        print("Using upload_large_folder for better handling of large files...")
        print()
        
        api.upload_large_folder(
            folder_path=CHECKPOINT_DIR,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            num_workers=4,  # Upload files in parallel
        )
        
        print()
        print("=" * 60)
        print("🎉 Upload completed successfully!")
        print("=" * 60)
        print()
        print(f"📍 Your checkpoint is now available at:")
        print(f"   https://huggingface.co/{REPO_ID}")
        print()
        print(f"🔗 To use in your Gradio app, use this path:")
        print(f'   checkpoint_path = "hf://datasets/{REPO_ID}"')
        print()
        print("📝 Note: It may take a few minutes for the files to be fully processed")
        print()
        
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        print()
        print("💡 Make sure you're logged in to Hugging Face:")
        print("   Run: huggingface-cli login")
        print("   Or set HF_TOKEN environment variable")

if __name__ == "__main__":
    upload_checkpoint()

