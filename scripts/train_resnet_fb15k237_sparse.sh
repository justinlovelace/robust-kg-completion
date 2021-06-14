#!/bin/bash
python train.py --strategy one_to_n --clip 1 --batch_size 64 --dropout 0.3 --feature_map_dropout 0.2 --input_dropout 0.2 --weight_decay 0.0001 --lr 0.001 --dataset FB15K_237_SPARSE --model PretrainedBertResNet --reshape_len 5 --resnet_block_depth 2 --resnet_num_blocks 3 --label_smoothing 0.1 --num_epochs 200
python evaluation.py --model PretrainedBertResNet --dataset FB15K_237_SPARSE