#!/bin/bash
dataset=$1
teacher_lambda=$2

python gen_triplet_dataset.py --model PretrainedBertResNet --dataset ${dataset} --model_dir pretrained_models/${dataset}/PretrainedBertResNet --topk 10
cd reranking/
python triplet_evaluation.py --model TripletTextBert --dataset ${dataset} --teacher_lambda ${teacher_lambda} --save_dir pretrained_reranking_models --action eval_mix --split valid
python triplet_evaluation.py --model TripletTextBert --dataset ${dataset} --teacher_lambda ${teacher_lambda} --save_dir pretrained_reranking_models --action eval_mix --split test