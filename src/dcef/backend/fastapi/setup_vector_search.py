#!/usr/bin/env python3
"""
Setup script for vector search dependencies.
This script checks if Qdrant is installed and running,
and ensures SentenceTransformer with the required model 
can be downloaded.
"""

import os
import pathlib
import requests
import importlib.util

# Get repo root path - move up from src/dcef/backend/fastapi to the repo root
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
VECTOR_INDEX_DIR = os.path.join(REPO_ROOT, "vector_index")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def check_qdrant():
    """Check if Qdrant is running"""
    try:
        response = requests.get("http://localhost:6333/collections")
        if response.status_code == 200:
            print("✅ Qdrant is running")
            return True
    except requests.exceptions.ConnectionError:
        pass
    
    print("❌ Qdrant is not running")
    print("\nTo install and run Qdrant:")
    print("1. Visit https://qdrant.tech/documentation/quick-start/")
    print("2. For Linux: Use Docker or download from GitHub")
    print("3. For Windows: Use Docker or binary")
    print("4. Start Qdrant and make sure it's running on port 6333")
    return False

def check_huggingface_deps():
    """Check if HuggingFace dependencies are installed and model is available"""
    # Check if required packages are installed
    try:
        # Check sentence-transformers
        st_available = importlib.util.find_spec("sentence_transformers")
        if not st_available:
            print("❌ sentence-transformers package is not installed")
            print("\nTo install the required packages:")
            print("pip install sentence-transformers>=2.2.2")
            return False
        
        print("✅ SentenceTransformer dependencies installed")
        
        # Test internet connectivity to HuggingFace Hub
        try:
            response = requests.head("https://huggingface.co/models", timeout=5)
            if response.status_code == 200:
                print(f"✅ HuggingFace Hub is accessible. Model '{EMBEDDING_MODEL}' will be downloaded on first use.")
                return True
            else:
                print("⚠️ Unable to verify connection to HuggingFace Hub")
                print(f"The model '{EMBEDDING_MODEL}' will be downloaded when first used if internet is available.")
                return True
        except requests.exceptions.RequestException:
            print("⚠️ Unable to connect to HuggingFace Hub")
            print(f"The model '{EMBEDDING_MODEL}' will be downloaded when first used if internet is available.")
            # Return True anyway since this is not a fatal error - model will be downloaded on first use
            return True
            
    except Exception as e:
        print(f"❌ Error checking HuggingFace dependencies: {e}")
        return False

def check_vector_index_dir():
    """Check if vector index directory exists and create it if not"""
    vector_dir = pathlib.Path(VECTOR_INDEX_DIR)
    if not vector_dir.exists():
        print(f"ℹ️ Creating vector index directory at: {VECTOR_INDEX_DIR}")
        try:
            vector_dir.mkdir(parents=True, exist_ok=True)
            # Also create embedding model cache directory
            model_cache = os.path.join(vector_dir, "embedding_model")
            pathlib.Path(model_cache).mkdir(parents=True, exist_ok=True)
            print("✅ Vector index directory created")
        except Exception as e:
            print(f"❌ Error creating vector index directory: {e}")
            return False
    else:
        print(f"✅ Vector index directory exists at: {VECTOR_INDEX_DIR}")
    return True

def check_dependencies():
    """Check if all dependencies are installed and running"""
    qdrant_ok = check_qdrant()
    huggingface_ok = check_huggingface_deps()
    vector_dir_ok = check_vector_index_dir()
    
    if qdrant_ok and huggingface_ok and vector_dir_ok:
        print("\n✅ All dependencies are running. You can now build and use the vector search index!")
        print(f"\nVector index will be stored at: {VECTOR_INDEX_DIR}")
        print("\nTo build the index for all guilds:")
        print("python -m src.vector_search.build_index")
        print("\nTo build the index for a specific guild:")
        print("python -m src.vector_search.build_index --guild_id <GUILD_ID>")
        return True
    else:
        print("\n❌ Some dependencies are missing or not running.")
        print("Please install and start the missing dependencies before building the vector index.")
        return False

if __name__ == "__main__":
    print("Checking vector search dependencies...\n")
    check_dependencies() 