#!/usr/bin/env python3
"""
Evaluate fine-tuned PaddleOCR-VL model on Manga109s test set.

Computes Character Error Rate (CER) and Exact Match accuracy.

Usage:
    python evaluate_model.py --model_path ./sft_output --num_samples 100
    python evaluate_model.py --model_path PaddlePaddle/PaddleOCR-VL  # baseline
"""

import argparse
import random
from pathlib import Path

import editdistance
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


def calculate_metrics(predictions: list, references: list) -> dict:
    """
    Calculate OCR evaluation metrics.
    
    Args:
        predictions: List of predicted strings
        references: List of ground truth strings
        
    Returns:
        Dictionary with CER, exact match rate, and counts
    """
    total_cer = 0
    exact_matches = 0
    
    for pred, ref in zip(predictions, references):
        # Character Error Rate
        cer = editdistance.eval(pred, ref) / max(len(ref), 1)
        total_cer += cer
        
        # Exact Match
        if pred.strip() == ref.strip():
            exact_matches += 1
    
    n = len(predictions)
    return {
        'cer': total_cer / n * 100,
        'exact_match': exact_matches / n * 100,
        'total_samples': n,
        'exact_match_count': exact_matches,
    }


def load_test_data(data_csv: Path, num_samples: int = None, split: str = 'test') -> list:
    """Load test samples from data.csv."""
    df = pd.read_csv(data_csv)
    df = df[df['split'] == split].reset_index(drop=True)
    
    samples = [
        {'image_path': row['crop_path'], 'text': row['text']}
        for _, row in df.iterrows()
    ]
    
    if num_samples and num_samples < len(samples):
        random.seed(42)
        samples = random.sample(samples, num_samples)
    
    print(f"Total {split} samples: {len(df)}, using: {len(samples)}")
    return samples


def evaluate_model(
    model_path: str,
    data_root: str,
    num_samples: int = 100,
    show_examples: int = 10
) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        model_path: Path to model (local or HuggingFace ID)
        data_root: Path to Manga109s data directory
        num_samples: Number of samples to evaluate
        show_examples: Number of example predictions to display
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 60)
    print("📊 PaddleOCR-VL Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Samples: {num_samples}")
    print()
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    print("✓ Model loaded")
    
    # Load test data
    data_csv = Path(data_root) / "data.csv"
    samples = load_test_data(data_csv, num_samples)
    
    # Run inference
    predictions = []
    references = []
    examples = []
    
    print("\nRunning inference...")
    for sample in tqdm(samples, desc="Evaluating"):
        image_path = Path(data_root) / sample['image_path']
        reference = sample['text']
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "OCR:"},
                ],
            }]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[[image]],
                return_tensors="pt",
                add_special_tokens=False,
            )
            inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
            
            input_len = inputs['input_ids'].shape[1]
            generated = outputs[0][input_len:]
            prediction = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            predictions.append(prediction)
            references.append(reference)
            
            if len(examples) < show_examples:
                examples.append({
                    'reference': reference,
                    'prediction': prediction,
                    'match': prediction.strip() == reference.strip(),
                })
                
        except Exception as e:
            print(f"Error: {image_path}: {e}")
            predictions.append("")
            references.append(reference)
    
    # Calculate and display metrics
    metrics = calculate_metrics(predictions, references)
    
    print()
    print("=" * 60)
    print("📈 Results")
    print("=" * 60)
    print(f"  Samples: {metrics['total_samples']}")
    print(f"  Exact Match: {metrics['exact_match']:.1f}% ({metrics['exact_match_count']}/{metrics['total_samples']})")
    print(f"  Character Error Rate (CER): {metrics['cer']:.2f}%")
    
    print()
    print("=" * 60)
    print("📝 Example Predictions")
    print("=" * 60)
    for i, ex in enumerate(examples):
        status = "✓" if ex['match'] else "✗"
        print(f"\nExample {i+1} [{status}]:")
        print(f"  Ground truth: {ex['reference']}")
        print(f"  Prediction:   {ex['prediction']}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate PaddleOCR-VL model")
    parser.add_argument("--model_path", default="./sft_output", 
                        help="Path to model (local or HuggingFace ID)")
    parser.add_argument("--data_root", default="./Manga109s_released_2023_12_07",
                        help="Path to Manga109s data directory")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--show_examples", type=int, default=10,
                        help="Number of examples to display")
    args = parser.parse_args()
    
    evaluate_model(
        args.model_path,
        args.data_root,
        args.num_samples,
        args.show_examples,
    )


if __name__ == "__main__":
    main()
