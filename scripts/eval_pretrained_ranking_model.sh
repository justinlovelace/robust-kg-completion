#!/bin/bash
dataset=$1

python evaluation.py --model PretrainedBertResNet --dataset ${dataset} --save_dir pretrained_models