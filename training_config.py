"""
Training Configuration for HERA Medical Chat
Sequential Fine-tuning Setup for Gemma-2-2b
"""

import torch
from transformers import TrainingArguments
from peft import LoraConfig
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration for HERA fine-tuning"""
    base_model: str = "google/gemma-2-2b-it"
    model_name: str = "HERA-Medical-Chat"
    cache_dir: str = "./cache"
    
    # Hardware optimization
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = "sdpa"  # or "eager" for older GPUs

@dataclass
class QuantizationConfig:
    """4-bit quantization configuration"""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_use_double_quant: bool = True

@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass 
class Stage1Config:
    """Stage 1: Medical Knowledge Training (PubMedQA)"""
    dataset_name: str = "qiaojin/PubMedQA"
    dataset_config: str = "pqa_artificial"
    num_samples: int = 3000
    
    # Training arguments
    output_dir: str = "./stage1_medqa"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 500
    logging_steps: int = 1
    
    # Optimization
    optim: str = "paged_adamw_32bit"
    fp16: bool = False
    bf16: bool = False
    group_by_length: bool = True
    
    # Monitoring
    report_to: str = "wandb"
    run_name: str = "HERA-Stage1-MedQA"

@dataclass
class Stage2Config:
    """Stage 2: Healthcare Conversations (HealthCareMagic)"""
    dataset_name: str = "HealthCareMagic"  # Custom dataset
    
    # Training arguments  
    output_dir: str = "./stage2_healthcare"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 1
    learning_rate: float = 1e-4  # Lower LR for stage 2
    warmup_steps: int = 5
    eval_steps: int = 100
    save_steps: int = 250
    logging_steps: int = 1
    
    # Optimization
    optim: str = "paged_adamw_32bit"
    fp16: bool = False
    bf16: bool = False
    group_by_length: bool = True
    
    # Monitoring
    report_to: str = "wandb"
    run_name: str = "HERA-Stage2-Healthcare"

def get_training_args(stage_config) -> TrainingArguments:
    """Convert stage config to TrainingArguments"""
    return TrainingArguments(
        output_dir=stage_config.output_dir,
        per_device_train_batch_size=stage_config.per_device_train_batch_size,
        per_device_eval_batch_size=stage_config.per_device_eval_batch_size,
        gradient_accumulation_steps=stage_config.gradient_accumulation_steps,
        num_train_epochs=stage_config.num_train_epochs,
        learning_rate=stage_config.learning_rate,
        warmup_steps=stage_config.warmup_steps,
        eval_strategy="steps",
        eval_steps=stage_config.eval_steps,
        save_steps=stage_config.save_steps,
        logging_steps=stage_config.logging_steps,
        optim=stage_config.optim,
        fp16=stage_config.fp16,
        bf16=stage_config.bf16,
        group_by_length=stage_config.group_by_length,
        report_to=stage_config.report_to,
        run_name=stage_config.run_name,
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )

def get_lora_config() -> LoraConfig:
    """Get LoRA configuration"""
    config = LoRAConfig()
    return LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=config.task_type,
        target_modules=config.target_modules
    )

# System prompts for different stages
STAGE1_SYSTEM_PROMPT = "You are a medical expert. Answer the question based on the provided context."

STAGE2_SYSTEM_PROMPT = """You are a helpful medical assistant. Provide informative and caring responses to health-related questions. Always recommend consulting healthcare professionals for serious concerns."""

# Usage example
if __name__ == "__main__":
    # Initialize configurations
    model_config = ModelConfig()
    stage1_config = Stage1Config()
    stage2_config = Stage2Config()
    
    print("HERA Training Configuration")
    print(f"Base Model: {model_config.base_model}")
    print(f"Stage 1 Dataset: {stage1_config.dataset_name}")
    print(f"Stage 2 Dataset: {stage2_config.dataset_name}")
    print(f"LoRA Rank: {LoRAConfig().r}")
