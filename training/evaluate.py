import torch
import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from data.dataset import get_batch

@torch.no_grad()
def estimate_loss(model, eval_iters, seq_len, batch_size, device):
    """
    Evaluates the model loss over a few batches to get an accurate estimate
    of both training and validation set performance.
    """
    out = {}
    model.eval()
    
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, seq_len, batch_size, device)
            
            if device == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits, loss = model(X, Y)
            else:
                logits, loss = model(X, Y)
                
            losses[k] = loss.item()
            
        out[split] = losses.mean().item()
        
    model.train()
    return out
