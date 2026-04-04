import os
import math
import time
import argparse
import torch
from torch.optim import AdamW

import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from model.config import ModelArgs
from model.transformer import TransformerLM
from data.dataset import get_batch
from training.evaluate import estimate_loss

def train(max_iters=5000, batch_size=8, learning_rate=3e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    eval_interval = 250
    eval_iters = 100
    warmup_iters = 100
    
    print(f"Initializing model on {device}...")
    
    config = ModelArgs()
    # If tuning for specific memory/speed, alter config here
    # config.n_layers = 4
    # config.n_heads = 4
    
    model = TransformerLM(config)
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M")
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-1)
    
    # Cosine scheduler with warmup
    def lr_lambda(current_step: int):
        if current_step < warmup_iters:
            return float(current_step) / float(max(1, warmup_iters))
        progress = float(current_step - warmup_iters) / float(max(1, max_iters - warmup_iters))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"Starting training loop for {max_iters} iterations...")
    best_val_loss = float('inf')
    
    t0 = time.time()
    for iter_num in range(max_iters):
        
        # Evaluate loss on train/val sets occasionally and save checkpoint
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model, eval_iters, config.max_seq_len, batch_size, device)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}")
            
            if losses['valid'] < best_val_loss:
                best_val_loss = losses['valid']
                os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
                ckpt_path = os.path.join(base_dir, "checkpoints", "model.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
                
        # Get batch
        X, Y = get_batch('train', config.max_seq_len, batch_size, device)
        
        # Forward pass
        if device == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits, loss = model(X, Y)
        else:
            logits, loss = model(X, Y)
            
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        if iter_num % 100 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num}: loss {loss.item():.4f}, dt {dt*1000:.2f}ms, lr {scheduler.get_last_lr()[0]:.2e}")

    print("Training Complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train custom LLM")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    
    train(max_iters=args.iters, batch_size=args.batch_size, learning_rate=args.lr)
