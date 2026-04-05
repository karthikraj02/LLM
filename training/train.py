import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

def train(model_id, data_path, output_dir, max_iters, batch_size, learning_rate):
    print(f"Loading tokenizer and model: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Setup QLoRA (4-bit quantization) for consumer GPUs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    # Prepare Model for LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    print(f"Trainable Parameters for LoRA:")
    model.print_trainable_parameters()
    
    # Load Dataset
    try:
        dataset = load_dataset("json", data_files={"train": data_path})["train"]
    except:
        print("Data load failed, falling back to timdettmers/openassistant-guanaco")
        dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    
    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=200,
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=True, 
        max_grad_norm=0.3,
        max_steps=max_iters,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none"
    )
    
    # Initialize Supervised Fine-Tuning Trainer (TRL)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    print("Starting QLoRA Fine-tuning...")
    trainer.train()
    
    print(f"Saving fine-tuned model adapters to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tune Open Source LLM")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="./lora-adapters")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    # Ignored args to be compatible with run.py
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--use_deepspeed", type=str, default="False")
    
    args = parser.parse_args()
    
    train(
        model_id=args.model_id,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )