#!/bin/bash


export CONFIG_PATH=./core/tiny_munit.yaml

export OUTPUT_PATH=./results-mnist

export CUDA_VISIBLE_DEVICES=0
python3 train_munit.py --config $CONFIG_PATH --output_path $OUTPUT_PATH