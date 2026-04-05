from dataclasses import dataclass

@dataclass
class ModelArgs:
    # Model dimensions
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    
    # Vocabulary and sequence length
    vocab_size: int = 16384  # Target BPE vocab size
    max_seq_len: int = 512
    
    # Regularization
    dropout: float = 0.1
    
    # Special tokens
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
