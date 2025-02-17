#!/bin/sh

python ./src/run_closed.py \
    --model_id gpt-3.5-turbo

python ./src/run_closed.py \
    --model_id gpt-3.5-turbo \
    --cot

python ./src/run_closed.py \
    --model_id gpt-3.5-turbo \
    --ps

python ./src/run_closed.py \
    --model_id gpt-3.5-turbo \
    --selfask