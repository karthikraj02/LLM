import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Custom LLM Toolkit")
    parser.add_argument("--train", action="store_true", help="Run the full pipeline: download -> preprocess -> train")
    parser.add_argument("--api", action="store_true", help="Start the FastAPI server")
    parser.add_argument("--cli", action="store_true", help="Start the interactive CLI")
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
        
    if args.train:
        print("\n=== 1. Downloading Dataset ===")
        subprocess.run([sys.executable, "data/download.py"])
        print("\n=== 2. Preprocessing & Tokenizing ===")
        subprocess.run([sys.executable, "data/preprocess.py"])
        print("\n=== 3. Training Model ===")
        subprocess.run([sys.executable, "training/train.py", "--iters", "5000", "--batch_size", "8"])
        
    if args.api:
        print("\n=== Starting API Server ===")
        print("To view Chat UI, open frontend/index.html in your web browser.")
        subprocess.run([sys.executable, "api/server.py"])
        
    if args.cli:
        print("\n=== Starting Interactive CLI ===")
        subprocess.run([sys.executable, "inference/cli.py"])

if __name__ == "__main__":
    main()
