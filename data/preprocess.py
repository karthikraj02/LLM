import os
import torch
import sys

# Add root project dir to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from model.tokenizer import BPETokenizer

def preprocess(vocab_size=4096, dataset_name="wikitext-2"):
    raw_dir = os.path.join(base_dir, "data", "raw", f"{dataset_name}-raw")
    train_path = os.path.join(raw_dir, "wiki.train.raw")
    valid_path = os.path.join(raw_dir, "wiki.valid.raw")
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Did you run download.py?")
        return

    print("Loading text data...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_text = f.read()

    # Clean text: remove headers starting and ending with '='
    train_text = "\n".join([line for line in train_text.splitlines() if line.strip() and not line.strip().startswith("=")])
    valid_text = "\n".join([line for line in valid_text.splitlines() if line.strip() and not line.strip().startswith("=")])

    print("Training tokenizer...")
    tokenizer = BPETokenizer()
    
    # The new HuggingFace Rust tokenizer is incredibly fast, so we train on EVERYTHING!
    print(f"Training on the full training set...")
    tokenizer.train([train_text], vocab_size=vocab_size, show_progress=True)
    
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    tokenizer.save(os.path.join(ckpt_dir, "tokenizer.json"))

    print("Tokenizing train dataset...")
    train_ids = tokenizer.encode(train_text)
    
    print("Tokenizing valid dataset...")
    valid_ids = tokenizer.encode(valid_text)

    proc_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    
    print(f"Train set tokens: {len(train_ids)}")
    print(f"Valid set tokens: {len(valid_ids)}")
    
    torch.save(torch.tensor(train_ids, dtype=torch.long), os.path.join(proc_dir, "train.pt"))
    torch.save(torch.tensor(valid_ids, dtype=torch.long), os.path.join(proc_dir, "valid.pt"))
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess(vocab_size=4096)
