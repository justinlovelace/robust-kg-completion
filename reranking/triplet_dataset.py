import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import argparse
import os
import sys
from TRIPLET_CONSTANTS import COLUMN_NAMES
import transformers
import triplet_utils


class KbTextDataset(Dataset):
    def __init__(self, args):
        
        self.args = args
        csv_file = os.path.join(args.teacher_folder, 'triplet_datasets', f'df_train_{args.topk}triplets.csv')
        e1_col, rel_col, e2_col = COLUMN_NAMES[args.dataset]
        print(f'Loading dataset from {csv_file}')
        df_kg = pd.read_csv(csv_file, converters={e2_col: eval, 'logits': eval, 'scores': eval, 'labels': eval})

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_model)
        print(len(self.tokenizer))
        self.tokenizer.add_tokens([f'[REL_{i}]' for i in range(args.num_relations)], special_tokens=True)
        print(len(self.tokenizer))
        args.vocab_size = len(self.tokenizer)
        self.gen_softmax_normalized_data(df_kg)


    def gen_softmax_normalized_data(self, df_kg):
        e1_col, rel_col, e2_col = COLUMN_NAMES[self.args.dataset]
        e2s = np.array(df_kg[e2_col].tolist(), dtype=np.int64)    
        queries = np.repeat(df_kg[[e1_col, rel_col]].to_numpy(np.int64), e2s.shape[1], axis=0)
        
        self.triplets = np.concatenate((queries, e2s.reshape(-1,1)), axis=1)

        self.ent2id, self.ent2name, self.rel2id = triplet_utils.get_kg_dicts(self.args)
        self.id2ent = {v: k for k,v in self.ent2id.items()}
        
        # self.ent2id, self.ent2name, self.rel2id = triplet_utils.get_kg_dicts(self.args)
        # self.id2ent = {v: k for k,v in self.ent2id.items()}
        # head_strings = [' '.join((f'[REL_{r}]', self.ent2name[self.id2ent[e1]])) for e1, r in zip(
        #     self.triplets[:,0], self.triplets[:,1])]
        # tail_strings = [' '.join((f'[REL_{r}]', self.ent2name[self.id2ent[e2]])) for e2, r in zip(
        #     self.triplets[:,2], self.triplets[:,1])]
        # self.encodings = self.tokenizer(text=head_strings, text_pair=tail_strings, is_split_into_words=False, padding=True, truncation=True, max_length=32 if 'FB' not in self.args.dataset else 256)

        softmax = torch.nn.Softmax(dim=1)

        # unnormalized_scores = np.array([score for scores_list in triplets['scores'] for score in scores_list])
        
        teacher_labels = softmax(torch.tensor(df_kg['logits'].tolist(), dtype=torch.float32)/self.args.temperature).numpy().flatten()
        teacher_labels *= self.args.teacher_lambda
        print(teacher_labels.shape)
        print(teacher_labels[0])
        self.gold_labels = np.array(df_kg['labels'].tolist(), dtype=np.float32).flatten()
        self.labels = self.gold_labels*(1-self.args.teacher_lambda)
        self.labels += teacher_labels

        assert len(self.triplets) == self.labels.size
        # print(unnormalized_scores)

        # print(self.scores)
        # print(self.scores.shape)
        # sys.exit()


    def __getitem__(self, idx):
        triple = self.triplets[idx]
        head_strings = ' '.join((f'[REL_{triple[1]}]', self.ent2name[self.id2ent[triple[0]]]))
        tail_strings = ' '.join((f'[REL_{triple[1]}]', self.ent2name[self.id2ent[triple[2]]]))
        encodings = self.tokenizer(text=head_strings, text_pair=tail_strings, is_split_into_words=False, padding='max_length', truncation=True, max_length=32)
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        return item, self.labels[idx]  

    def __len__(self):
        return len(self.labels)


class KbTextEvalGenerator(Dataset):
    def __init__(self, eval_split: str, args: argparse.ArgumentParser):
        self.args = args
        csv_file = os.path.join(args.teacher_folder, 'triplet_datasets', f'df_{eval_split}_{args.topk}triplets.csv')
        e1_col, rel_col, e2_col = COLUMN_NAMES[args.dataset]
        print(f'Loading dataset from {csv_file}')
        df_kg = pd.read_csv(csv_file, converters={e2_col: eval, 'logits': eval, 'scores': eval, 'labels': eval})

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_model)
        print(len(self.tokenizer))
        self.tokenizer.add_tokens([f'[REL_{i}]' for i in range(args.num_relations)], special_tokens=True)
        print(args.num_relations)
        print(len(self.tokenizer))
        args.vocab_size = len(self.tokenizer)

        self.gen_ranking_data(df_kg)

    def gen_ranking_data(self, df_kg):
        e1_col, rel_col, e2_col = COLUMN_NAMES[self.args.dataset]
        e2s = np.array(df_kg[e2_col].tolist(), dtype=np.int64)    
        queries = np.repeat(df_kg[[e1_col, rel_col]].to_numpy(np.int64), e2s.shape[1], axis=0)
        
        triplets = np.concatenate((queries, e2s.reshape(-1,1)), axis=1)

        softmax = torch.nn.Softmax(dim=1)
        self.teacher_labels = softmax(torch.tensor(df_kg['logits'].tolist(), dtype=torch.float32))

        self.ent2id, self.ent2name, self.rel2id = triplet_utils.get_kg_dicts(self.args)
        self.id2ent = {v: k for k,v in self.ent2id.items()}
        head_strings = [' '.join((f'[REL_{r}]', self.ent2name[self.id2ent[e1]])) for e1, r in zip(
            triplets[:,0], triplets[:,1])]
        tail_strings = [' '.join((f'[REL_{r}]', self.ent2name[self.id2ent[e2]])) for e2, r in zip(
            triplets[:,2], triplets[:,1])]
        print(self.tokenizer.tokenize(head_strings[0], tail_strings[0], is_split_into_words=False, padding=True, truncation=True, max_length=32))
        # sys.exit()
        self.encodings = self.tokenizer(text=head_strings, text_pair=tail_strings, is_split_into_words=False, padding='max_length', truncation=True, max_length=32)

        self.labels = np.array(df_kg['labels'].tolist(), dtype=np.float32)
        print(self.labels.shape)


        

        self.baseline_metrics = self.get_baseline()
        print(len(self.encodings['input_ids']))
        print(self.labels.shape)
        print(self.labels.size)
        self.unrolled_triplets = triplets
        self.triplets = triplets.reshape((-1, self.args.topk, 3))
        print(self.triplets.shape)
        
        assert len(self.encodings['input_ids']) == self.labels.size
        assert self.triplets.shape[0] == self.labels.shape[0]
    
    def get_baseline_ranks(self):
        ranks = np.nonzero(self.labels == 1)[1] + 1
        print(ranks)
        return ranks

    def get_baseline(self):

        ranks = np.nonzero(self.labels == 1)[1] + 1
        print(ranks)
        print(ranks.shape)
        metrics = {}
        metrics['MR'] = np.mean(ranks).item()
        metrics['MRR'] = np.mean(1. / ranks).item()
        metrics['H@1'] = np.mean(np.where(ranks < 2, 1, 0)).item()
        metrics['H@3'] = np.mean(np.where(ranks < 4, 1, 0)).item()
        metrics['H@5'] = np.mean(np.where(ranks < 6, 1, 0)).item()
        metrics['H@10'] = np.mean(np.where(ranks < 11, 1, 0)).item()
        metrics['H@100'] = np.mean(np.where(ranks < 101, 1, 0)).item()
        metrics['H@1000'] = np.mean(np.where(ranks < 1001, 1, 0)).item()
        metrics['H@10000'] = np.mean(np.where(ranks < 10001, 1, 0)).item() 

        return metrics

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start_idx = idx*self.args.topk
        # head_strings = [' '.join((f'[REL_{r}]', self.ent2name[self.id2ent[e1]])) for e1, r in zip(
        #     self.unrolled_triplets[start_idx:start_idx+self.args.topk:,0], self.unrolled_triplets[start_idx:start_idx+self.args.topk,1])]
        # tail_strings = [' '.join((f'[REL_{r}]', self.ent2name[self.id2ent[e2]])) for e2, r in zip(
        #     self.unrolled_triplets[start_idx:start_idx+self.args.topk:,2], self.unrolled_triplets[start_idx:start_idx+self.args.topk,1])]

        # item = self.tokenizer(text=head_strings, text_pair=tail_strings, is_split_into_words=False, padding='max_length', truncation=True, max_length=32)
        item = {key: torch.tensor(val[start_idx:start_idx+self.args.topk]) for key, val in self.encodings.items()}
        return item, self.labels[idx]


