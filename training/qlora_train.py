"""
QLoRA Fine-tuning Script
Uses bitsandbytes 4-bit quantization + PEFT LoRA adapters + TRL SFTTrainer
to fine-tune a pre-trained causal LM on a small instruction dataset.

Default model: HuggingFaceTB/SmolLM-135M (tiny, no gated access needed)
Swap MODEL_ID to e.g. "microsoft/Phi-3-mini-4k-instruct" or
"TinyLlama/TinyLlama-1.1B-Chat-v1.0" for a stronger baseline.
"""

import os
import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Defaults – override via CLI flags
# ---------------------------------------------------------------------------
MODEL_ID = "HuggingFaceTB/SmolLM-135M"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "qlora_adapter")


def build_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization config for bitsandbytes."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05) -> LoraConfig:
    """LoRA adapter configuration."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        # Target the attention projection matrices (common for most models)
        target_modules=["q_proj", "v_proj"],
    )


def format_instruction(example: dict) -> str:
    """Convert a dataset row into an instruction-following prompt."""
    instruction = example.get("instruction", "")
    response = example.get("output", example.get("response", ""))
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def run_training(
    model_id: str = MODEL_ID,
    output_dir: str = OUTPUT_DIR,
    dataset_name: str = "databricks/databricks-dolly-15k",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    print(f"\n{'='*60}")
    print(f"QLoRA Fine-tuning")
    print(f"  Base model  : {model_id}")
    print(f"  Dataset     : {dataset_name}")
    print(f"  Output dir  : {output_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # 2. Model in 4-bit
    # ------------------------------------------------------------------
    cuda_available = torch.cuda.is_available()
    print(f"Loading model (4-bit quantization: {cuda_available})...")

    model_kwargs = dict(trust_remote_code=True)
    if cuda_available:
        model_kwargs["quantization_config"] = build_bnb_config()
        model_kwargs["device_map"] = "auto"
    else:
        # CPU fallback – load in full precision (slow, for debugging only)
        model_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False  # required for gradient checkpointing

    # ------------------------------------------------------------------
    # 3. Apply LoRA
    # ------------------------------------------------------------------
    print("Applying LoRA adapters...")
    lora_cfg = build_lora_config(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 4. Dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset: {dataset_name} ...")
    raw_dataset = load_dataset(dataset_name, split="train")

    # ------------------------------------------------------------------
    # 5. Training arguments
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=cuda_available,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        max_seq_length=max_seq_length,
    )

    # ------------------------------------------------------------------
    # 6. SFTTrainer
    # ------------------------------------------------------------------
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=raw_dataset,
        formatting_func=format_instruction,
        args=training_args,
    )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print("Starting training...\n")
    trainer.train()

    # ------------------------------------------------------------------
    # 8. Save adapter + tokenizer
    # ------------------------------------------------------------------
    print(f"\nSaving LoRA adapter to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning with PEFT + bitsandbytes")
    parser.add_argument("--model-id", default=MODEL_ID,
                        help="HuggingFace model ID (default: HuggingFaceTB/SmolLM-135M)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory to save the LoRA adapter")
    parser.add_argument("--dataset", default="databricks/databricks-dolly-15k",
                        help="HuggingFace dataset name")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    run_training(
        model_id=args.model_id,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
