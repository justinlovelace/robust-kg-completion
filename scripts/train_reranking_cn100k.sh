#!/bin/bash
dataset="CN100K"
teacher_lambda="1.0"

python gen_triplet_dataset.py --model PretrainedBertResNet --dataset ${dataset} --model_dir saved_models/${dataset}/PretrainedBertResNet --topk 10
cd reranking/
python triplet_train.py --clip 1 --batch_size 128 --weight_decay 0.01 --lr 0.00003 --dataset ${dataset} --model TripletTextBert --teacher_lambda ${teacher_lambda} --num_epochs 10 --teacher_folder ../saved_models/${dataset}/PretrainedBertResNet --topk 10 --temperature 1 --dropout 0.3
python triplet_evaluation.py --model TripletTextBert --dataset ${dataset} --teacher_lambda ${teacher_lambda} --action eval_mix --split valid
python triplet_evaluation.py --model TripletTextBert --dataset ${dataset} --teacher_lambda ${teacher_lambda} --action eval_mix --split test