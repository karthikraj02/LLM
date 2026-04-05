import torch
import os
import sys

# Setup relative paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from model.config import ModelArgs
from model.transformer import TransformerLM
from model.tokenizer import BPETokenizer

class LLMPipeline:
    def __init__(self, model_path=None, tokenizer_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load configurations
        self.config = ModelArgs()
        self.model = TransformerLM(self.config)
        
        # Resolve target paths
        if model_path is None:
            model_path = os.path.join(base_dir, "checkpoints", "model.pt")
            
        if tokenizer_path is None:
            tokenizer_path = os.path.join(base_dir, "checkpoints", "tokenizer.json")
            
        # Initialize tokenzier first
        self.tokenizer = BPETokenizer()
        if os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)
            # Update vocab_size config to match tokenzier
            self.model.config.vocab_size = len(self.tokenizer.vocab)
        else:
            print(f"Warning: Tokenizer not found at {tokenizer_path}. Using default byte semantics.")
            
        # Load model weights
        if os.path.exists(model_path):
            current_vocab_size = self.model.config.vocab_size
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint)
            
            # Phase 7: Apply dynamic quantization for CPU to speed up inference by ~2-3x
            if self.device == 'cpu':
                print("Applying dynamic quantization to Linear layers for faster CPU inference...")
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
        else:
            print(f"Warning: No model checkpoint found at {model_path}. Expect garbage output.")
            
        self.model.to(self.device)
        self.model.eval()

        from inference.safety import ToxicityFilter
        self.safety_filter = ToxicityFilter()

    def generate(self, prompt: str, max_new_tokens=50, temperature=0.8, top_k=40, top_p=0.9):
        # Phase 7: Basic Toxicity filtering on input
        if self.safety_filter.is_toxic(prompt):
            return "I cannot fulfill this request due to safety guidelines."
            
        # Text to token ids
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Raw generation
        output_tensor = self.model.generate(
            input_tensor, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
        )
        
        # Convert batched sequence (batch=1) to list
        output_ids = output_tensor[0].tolist()
        
        # Extract only the freshly generated tokens
        generated_ids = output_ids[len(input_ids):]
        
        # Token ids to text
        generated_text = self.tokenizer.decode(generated_ids)
        
        # Phase 7: Output safety check
        return self.safety_filter.clean(generated_text)

