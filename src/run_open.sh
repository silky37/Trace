#!/bin/sh

n=1

CUDA_VISIBLE_DEVICES=$n python ./src/run_open.py \
    --model_id meta-llama/Llama-3.2-1B-Instruct

CUDA_VISIBLE_DEVICES=$n python ./src/run_open.py \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --cot

CUDA_VISIBLE_DEVICES=$n python ./src/run_open.py \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --ps

CUDA_VISIBLE_DEVICES=$n python ./src/run_open.py \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --selfask