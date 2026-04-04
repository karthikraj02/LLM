import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Custom LLM Toolkit")
    parser.add_argument("--train", action="store_true",
                        help="Run the full custom-model pipeline: download -> preprocess -> train")
    parser.add_argument("--train-qlora", action="store_true",
                        help="Fine-tune a pre-trained HuggingFace model with QLoRA")
    parser.add_argument("--api", action="store_true",
                        help="Start the FastAPI inference server (vLLM or transformers backend)")
    parser.add_argument("--cli", action="store_true",
                        help="Start the interactive CLI")
    # Forward unknown args to sub-scripts (e.g. --model-id, --epochs, ...)
    args, extra = parser.parse_known_args()

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

    if args.train_qlora:
        print("\n=== QLoRA Fine-tuning ===")
        cmd = [sys.executable, "training/qlora_train.py"] + extra
        subprocess.run(cmd)

    if args.api:
        print("\n=== Starting API Server ===")
        print("Endpoint: http://0.0.0.0:8000/generate  (POST)")
        print("Health:   http://0.0.0.0:8000/health    (GET)")
        print("Docs:     http://0.0.0.0:8000/docs       (GET)")
        subprocess.run([sys.executable, "api/server.py"])

    if args.cli:
        print("\n=== Starting Interactive CLI ===")
        subprocess.run([sys.executable, "inference/cli.py"])

if __name__ == "__main__":
    main()
