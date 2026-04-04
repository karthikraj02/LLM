import os
from typing import List, Union
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class BPETokenizer:
    def __init__(self):
        # Wrap the blistering fast HuggingFace tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()
        
        self.pad_id = None
        self.bos_id = None
        self.eos_id = None
        self.unk_id = None

    def train(self, texts: List[str], vocab_size: int, show_progress: bool = True):
        # Configure the Rust trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=show_progress
        )
        
        # Train! This relies on the ultra-fast Rust implementation
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        self._update_special_ids()

    def _update_special_ids(self):
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        # Automatically ignores special tokens
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, file_path: str):
        self.tokenizer.save(file_path)
        
    def load(self, file_path: str):
        self.tokenizer = Tokenizer.from_file(file_path)
        self._update_special_ids()

    @property
    def vocab(self):
        # Provides backwards compatibility for scripts that do `len(tokenizer.vocab)`
        return self.tokenizer.get_vocab()
