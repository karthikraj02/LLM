import torch
import os

def get_batch(split: str, seq_len: int, batch_size: int, device: str = 'cpu') -> tuple:
    """
    Generates a small batch of data of inputs x and targets y.
    Loading the entire dataset into RAM is fine for WikiText-2.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", f"{split}.pt")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed file {data_path} not found. Ensure preprocess.py was run.")
        
    data = torch.load(data_path)
    
    # Randomly select starting indices for the batch
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    
    # x is sequence of tokens, y is same sequence shifted by 1 to the right
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    
    # Move batch to device (GPU/CPU)
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    
    return x, y
