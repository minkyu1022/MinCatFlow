#!/bin/bash

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python src/run.py \
    expname=mincat_gen_dfm_smI \
    train.pl_trainer.devices=4 \
    train.pl_trainer.strategy=ddp \
    model.flow_model_args.dng=true \
    train.pl_trainer.strategy=ddp_find_unused_parameters_true
