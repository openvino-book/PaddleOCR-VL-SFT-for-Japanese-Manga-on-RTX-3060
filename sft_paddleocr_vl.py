"""
Supervised Fine-Tuning (SFT) script for PaddleOCR-VL on Manga109s dataset.

This script fine-tunes PaddleOCR-VL for Japanese manga OCR using BF16 precision.
Optimized for RTX 3060 (12GB VRAM) but works on any GPU supporting BF16.

Usage:
    python sft_paddleocr_vl.py --help
    bash train.sh
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from custom_collator import CustomDataCollatorForVisionLanguageModeling
from ocr_dataset import MangaDataset


class BF16Trainer(Trainer):
    """
    Custom Trainer using BF16 autocast without GradScaler.
    
    RTX 3060 and newer GPUs support BF16 which has better numerical stability 
    than FP16 and doesn't require loss scaling. This trainer wraps the training
    and prediction steps with torch.amp.autocast for BF16 computation.
    """
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to use BF16 autocast."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        return loss.detach()
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction_step to use BF16 autocast during evaluation."""
        model.eval()
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = self.compute_loss(model, inputs)
                return (loss, None, None)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_path: str = field(
        default="PaddlePaddle/PaddleOCR-VL",
        metadata={"help": "Path to the PaddleOCR-VL model (HuggingFace ID or local path)"}
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable Flash Attention 2 for faster training. "
                "Requires flash-attn package and A100/H100 GPU. "
                "Default False for RTX 3060 compatibility."
            )
        },
    )


@dataclass
class DataArguments:
    """Arguments for dataset configuration."""

    split: str = field(
        default="train", 
        metadata={"help": "Dataset split for training: 'train'"}
    )
    eval_split: str = field(
        default="test",
        metadata={"help": "Dataset split for evaluation: 'test'"},
    )
    max_length: int = field(
        default=1536,
        metadata={
            "help": (
                "Maximum sequence length (image + text tokens). "
                "PaddleOCR-VL images use 400-2000+ tokens depending on size."
            )
        },
    )
    eval_limit_size: Optional[int] = field(
        default=1000,
        metadata={"help": "Limit eval dataset size to reduce memory usage."},
    )
    skip_packages: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of synthetic data package IDs to skip"},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=8,
        metadata={"help": "Pad sequence length to multiple of this value for GPU efficiency."},
    )


def train():
    """Main training function."""

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Required for VL models
    training_args.remove_unused_columns = False
    training_args.prediction_loss_only = True  # Avoid OOM during evaluation

    # Load model in BF16
    print(f"Loading model from {model_args.model_path}...")
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": DEVICE,
    }

    if model_args.use_flash_attention_2:
        print("🚀 Flash Attention 2 enabled")
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_args.model_path, **model_kwargs)
    print(f"✓ Model loaded in {next(model.parameters()).dtype}")

    processor = AutoProcessor.from_pretrained(
        model_args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    # Parse skip_packages
    skip_packages = None
    if data_args.skip_packages:
        skip_packages = [int(x.strip()) for x in data_args.skip_packages.split(",")]

    # Load datasets
    print(f"\nLoading {data_args.split} dataset...")
    train_dataset = MangaDataset(
        split=data_args.split,
        skip_packages=skip_packages,
    )
    print(f"Training dataset size: {len(train_dataset)}")

    print(f"\nLoading {data_args.eval_split} dataset...")
    eval_dataset = MangaDataset(
        split=data_args.eval_split,
        limit_size=data_args.eval_limit_size,
        augment=False,
        skip_packages=skip_packages,
    )
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    # Data collator
    collator = CustomDataCollatorForVisionLanguageModeling(
        processor,
        max_length=data_args.max_length,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
    )

    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    # Initialize trainer
    print("Initializing BF16Trainer...")
    trainer = BF16Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    
    checkpoint = training_args.resume_from_checkpoint
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)

    # Save model
    print(f"\nSaving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    print("✓ Training complete!")


if __name__ == "__main__":
    train()
