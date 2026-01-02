#!/bin/bash

CUDA_VISIBLE_DEVICES=5,6,7 python save_valid_samples.py \
    --checkpoint /home/minkyu/MinCatFlow/outputs/ads_mincat_gen/last.ckpt \
    --val_lmdb_path /home/minkyu/MinCatFlow/dataset_per_adsorbate/val_id/val_id_H-H-C-C-O-O_subset.lmdb \
    --output_dir unrelaxed_samples/de_novo_generation/C2H2O2/1/ \
    --num_samples 1 \
    --sampling_steps 50 \
    --batch_size 64 \
    --num_workers 32 \
    --use_ddp \
    --gpus 3
