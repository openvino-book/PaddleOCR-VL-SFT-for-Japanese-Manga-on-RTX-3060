#!/bin/bash
# =============================================================================
# PaddleOCR-VL Fine-tuning Script for RTX 3060 (12GB VRAM)
# 
# Uses BF16 mixed precision training via custom BF16Trainer.
# Training time: ~27 hours for 3 epochs on RTX 3060.
# =============================================================================

python sft_paddleocr_vl.py \
    --run_name "PaddleOCR-VL-Manga109s" \
    --model_path PaddlePaddle/PaddleOCR-VL \
    --split train \
    --max_length 1536 \
    --pad_to_multiple_of 8 \
    --output_dir ./sft_output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 500 \
    --per_device_eval_batch_size 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 3 \
    --dataloader_num_workers 2 \
    --gradient_checkpointing \
    --optim adamw_torch \
    --report_to none
