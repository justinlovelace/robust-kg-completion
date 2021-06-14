# Robust Knowledge Graph Completion with Stacked Convolutions and a Student Re-Ranking Network


This repository contains the implementation for our paper: 

**Robust Knowledge Graph Completion with Stacked Convolutions and a Student Re-Ranking Network** \
Justin Lovelace, Denis Newman-Griffis, Shikhar Vashishth, Jill Fain Lehman, and Carolyn Penstein Rosé \
Annual Meeting of the Association for Computational Linguistics and the International Joint Conference on Natural Language Processing
(**ACL-IJCNLP**) 2021

## Dependencies

Our work was performed with Python 3.8. The dependencies can be installed from `requirements.txt`.

## Data Preparation

- We conduct our work upon the existing FB15K-237 and CN100K datasets. We additionally developed the FB15K-237-Sparse and SNOMED-CT Core datasets for our work.
- Running `./scripts/prepare_datasets.sh` will unzip the dataset files and process them for use by our models.
- Because the SNOMED-CT Core dataset was derived from the UMLS, we cannot directly release the dataset files. See [here](snomed_ct_core.md) for full instructions for how to recreate the dataset.
- The BERT embeddings can be downloaded from [here](https://drive.google.com/drive/folders/1gfbZcJoay69BUzQLQku-qHB6IZ5zxLQw?usp=sharing). The `bert_emb.pt` files should be stored in the corresponding dataset directories, e.g. `data/CN100K/bert_emb.pt`

## Training Ranking Models
We provide scripts to train our proposed ranking model, denoted as BERT-ResNet in our paper, for all four datasets. 

- FB15K-237: `./scripts/train_resnet_fb15k237.sh`
- FB15K-237-Sparse: `./scripts/train_resnet_fb15k237_sparse.sh`
- CN100K: `./scripts/train_resnet_cn100k.sh`
- SNOMED-CT Core `./scripts/train_resnet_snomed.sh`

## Training Re-Ranking Models
The re-ranking models can only be trained after the ranking model for the corresponding dataset has already finished training. First, download the BERT checkpoints used for our training from [here](https://drive.google.com/drive/folders/1BsxeWEtFpZuHD_bCQKsIy0zfwl7xi6Bq?usp=sharing). They should be unzipped and stored in `reranking/bert_ckpts`. A re-ranking model can then be trained with the provided scripts similarly to above.

- FB15K-237: `./scripts/train_reranking_fb15k237.sh`
- FB15K-237-Sparse: `./scripts/train_reranking_fb15k237_sparse.sh`
- CN100K: `./scripts/train_reranking_cn100k.sh`
- SNOMED-CT Core `./scripts/train_reranking_snomed.sh`

## Evaluating Pretrained Ranking Models
Pretrained ranking models can be downloaded from [here](https://drive.google.com/drive/folders/1q20hhUq20wt5OSbHbOWsvAiviFUZ8s8r?usp=sharing). After unzipping them in a `robust-kg-completion/pretrained_models` directory, they can be evaluated by running `./scripts/eval_pretrained_ranking_model.sh {DATASET}` where `{DATASET}` is one of `SNOMED_CT_CORE`, `FB15K_237`, `FB15K_237_SPARSE`, or `CN100K`.

## Evaluating Pretrained Re-Ranking Models 
Pretrained re-ranking models can be downloaded from [here](https://drive.google.com/drive/folders/1q20hhUq20wt5OSbHbOWsvAiviFUZ8s8r?usp=sharing). After unzipping them in a `robust-kg-completion/reranking/pretrained_reranking_models` directory, they can be evaluated by running the following commands.

- FB15K-237: `./scripts/eval_pretrained_reranking_model.sh FB15K_237 0.75`
- FB15K-237-Sparse: `./scripts/eval_pretrained_reranking_model.sh FB15K_237_SPARSE 0.75`
- CN100K: `./scripts/eval_pretrained_reranking_model.sh CN100K 1.0`
- SNOMED-CT Core `./scripts/eval_pretrained_reranking_model.sh SNOMED_CT_CORE 0.5`

