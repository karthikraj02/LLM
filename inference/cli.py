import argparse
import sys
import os

# Ensure imports work regardless of execution folder
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from inference.generate import LLMPipeline

def main():
    parser = argparse.ArgumentParser(description="Custom LLM Inference CLI")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum new tokens to generate")
    args = parser.parse_args()
    
    print("Loading LLM Pipeline...")
    try:
        pipeline = LLMPipeline()
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        return
        
    print("\n" + "="*50)
    print("Welcome to Custom LLM CLI!")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            prompt = input("Prompt> ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
            if not prompt.strip():
                continue
                
            print("Response> ", end="", flush=True)
            
            # Not streaming in this CLI (optional feature), just print final
            response = pipeline.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            print(response + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
