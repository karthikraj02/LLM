import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="High-Performance LLM Toolkit")
    parser.add_argument("--train", action="store_true", help="Run full pipeline with Distributed Data Parallel (DDP) / DeepSpeed")
    parser.add_argument("--api", action="store_true", help="Start the FastAPI server powered by vLLM for high-speed inference")
    parser.add_argument("--cli", action="store_true", help="Start the interactive CLI")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for training")
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
        
    if args.train:
        print("\n=== 1. Downloading & Scaling Dataset ===")
        subprocess.run([sys.executable, "data/download.py", "--streaming", "True"])
        
        print("\n=== 2. High-Speed Tokenization (Rust-based) ===")
        subprocess.run([sys.executable, "data/preprocess.py", "--num_workers", str(os.cpu_count())])
        
        print(f"\n=== 3. Distributed Training Model on {args.gpus} GPUs ===")
        if args.gpus > 1:
            cmd = [
                "torchrun", "--nproc_per_node", str(args.gpus),
                "training/train.py", 
                "--iters", "50000",
                "--batch_size", "32",
                "--mixed_precision", "bf16",
                "--gradient_accumulation", "4",
                "--use_deepspeed", "True"
            ]
        else:
            cmd = [
                sys.executable, "training/train.py", 
                "--iters", "50000", 
                "--batch_size", "8",
                "--mixed_precision", "bf16"
            ]
        subprocess.run(cmd)
        
    if args.api:
        print("\n=== Starting Ultra-Fast vLLM API Server ===")
        print("To view Chat UI, open frontend/index.html in your web browser.")
        os.environ["VLLM_USE_MODELSCOPE"] = "True" 
        subprocess.run([
            sys.executable, "api/server.py",
            "--engine", "vllm",
            "--quantization", "awq",
            "--gpu-memory-utilization", "0.90"
        ])
        
    if args.cli:
        print("\n=== Starting Interactive CLI ===")
        subprocess.run([
            sys.executable, "inference/cli.py",
            "--use_flash_attention_2", "True",
            "--compile", "True"
        ])

if __name__ == "__main__":
    main()