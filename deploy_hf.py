#!/usr/bin/env python3
"""
🚀 NAVADA 2.0 - Hugging Face Spaces Deployment Script
Deploy NAVADA to Hugging Face Spaces with Streamlit
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo # type: ignore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def deploy_to_hf_spaces():
    """Deploy NAVADA 2.0 to Hugging Face Spaces"""
    
    # Configuration
    HF_TOKEN = os.getenv("HF_TOKEN")
    REPO_NAME = "navada-2-0-ai-computer-vision"
    REPO_TYPE = "space"
    
    if not HF_TOKEN:
        print("❌ HF_TOKEN not found in .env file")
        return False
    
    print("Starting NAVADA 2.0 deployment to Hugging Face Spaces...")
    
    try:
        # Initialize HF API
        api = HfApi(token=HF_TOKEN)
        
        # Get username
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{REPO_NAME}"
        
        print(f"👤 Deploying as: {username}")
        print(f"📦 Repository: {repo_id}")
        
        # Create repository if it doesn't exist
        try:
            create_repo(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                token=HF_TOKEN,
                space_sdk="streamlit",
                private=False
            )
            print("✅ Repository created successfully!")
        except Exception as e:
            if "already exists" in str(e):
                print("📁 Repository already exists, updating...")
            else:
                print(f"❌ Error creating repository: {e}")
                return False
        
        # Files to upload
        files_to_upload = [
            "app.py",
            "requirements.txt",
            "README.md",
            "README_DEPLOYMENT.md", 
            "packages.txt",
            ".streamlit/config.toml"
        ]
        
        # Upload backend directory
        backend_files = [
            "backend/__init__.py",
            "backend/yolo.py",
            "backend/openai_client.py",
            "backend/face_detection.py",
            "backend/recognition.py",
            "backend/database.py"
        ]
        
        all_files = files_to_upload + backend_files
        
        print("📤 Uploading files...")
        
        for file_path in all_files:
            if os.path.exists(file_path):
                try:
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=file_path,
                        repo_id=repo_id,
                        repo_type=REPO_TYPE,
                        token=HF_TOKEN
                    )
                    print(f"✅ Uploaded: {file_path}")
                except Exception as e:
                    print(f"❌ Failed to upload {file_path}: {e}")
            else:
                print(f"⚠️  File not found: {file_path}")
        
        # Set up environment variables in the Space
        print("🔧 Setting up environment variables...")
        
        # Note: Environment variables need to be set manually in HF Spaces UI
        print("⚠️  IMPORTANT: You need to manually add these secrets in HF Spaces:")
        print("   1. Go to your Space settings")
        print("   2. Add secret: OPENAI_API_KEY = your_openai_key")
        
        space_url = f"https://huggingface.co/spaces/{repo_id}"
        print(f"🎉 Deployment completed!")
        print(f"🔗 Your NAVADA 2.0 app will be available at: {space_url}")
        print(f"⏳ It may take a few minutes to build and deploy...")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return False

if __name__ == "__main__":
    success = deploy_to_hf_spaces()
    if success:
        print("\n🚀 NAVADA 2.0 deployment initiated successfully!")
        print("📖 Check README_DEPLOYMENT.md for additional setup instructions")
    else:
        print("\n❌ Deployment failed. Please check the errors above.")