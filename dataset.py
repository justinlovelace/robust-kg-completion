import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from collections import defaultdict
import sys
from CONSTANTS import DATA_DIR, COLUMN_NAMES


class KbDataset(Dataset):
    def __init__(self, args: argparse.ArgumentParser):
        self.args = args
        csv_file = os.path.join(DATA_DIR[args.dataset], f'df_train.csv')
        e1_col, rel_col, e2_col = COLUMN_NAMES[args.dataset]
        print(f'Loading dataset from {csv_file}')
        df_kg = pd.read_csv(csv_file)
        self.pos_samples = df_kg[[e1_col,
                                  rel_col, e2_col]].to_numpy(np.int64)
        # print(self.pos_samples.shape)
        if args.strategy == 'one_to_n':
            self.gen_one_to_n_data(df_kg)
        elif args.strategy == 'k_to_n' or args.strategy == 'gen_triplets':
            self.gen_k_to_n_data(df_kg)
        elif args.strategy == 'softmax':
            self.gen_softmax_data(df_kg)
        else:
            raise NotImplementedError

    def gen_one_to_n_data(self, df_kg):
        e1_col, rel_col, e2_col = COLUMN_NAMES[self.args.dataset]
        self.data = df_kg[[
            e1_col, rel_col]].to_numpy(np.int64)
        self.labels = np.zeros(
            (len(self.data), self.args.num_entities), dtype=np.float32)
        self.labels[np.arange(start=0, stop=self.labels.shape[0],
                              step=1), df_kg[e2_col].to_numpy(np.int64)] = 1

    def gen_softmax_data(self, df_kg):
        e1_col, rel_col, e2_col = COLUMN_NAMES[self.args.dataset]
        self.data = df_kg[[
            e1_col, rel_col]].to_numpy(np.int64)
        self.labels = df_kg[e2_col].to_numpy(np.int64)

    def gen_k_to_n_data(self, df_kg):
        e1_col, rel_col, e2_col = COLUMN_NAMES[self.args.dataset]
        self.data = df_kg[[
            e1_col, rel_col]].drop_duplicates().to_numpy(np.int64)
        e2_lookup = defaultdict(set)
        for e1, r, e2 in zip(df_kg[e1_col], df_kg[rel_col], df_kg[e2_col]):
            e2_lookup[(e1, r)].add(e2)
        self.labels = np.zeros(
            (len(self.data), self.args.num_entities), dtype=np.float32)
        for idx, query in enumerate(self.data):
            e1, r = query[0], query[1]
            for e2 in e2_lookup[(e1, r)]:
                self.labels[idx, e2] = 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class KbEvalGenerator(Dataset):
    def __init__(self, eval_split: str, args: argparse.ArgumentParser):
        self.args = args
        pos_samples = defaultdict(set)
        splits = ['train', 'valid', 'test']
        e1_col, rel_col, e2_col = COLUMN_NAMES[args.dataset]
        assert eval_split in splits
        for spl in splits:
            csv_file = os.path.join(DATA_DIR[args.dataset], f'df_{spl}.csv')
            df_data = pd.read_csv(csv_file)
            if spl == eval_split:
                self.queries = df_data[[
                    e1_col, rel_col]].to_numpy(np.int64)
                e2_list = df_data[e2_col].tolist()
            for e1, r, e2 in zip(df_data[e1_col], df_data[rel_col], df_data[e2_col]):
                pos_samples[(e1, r)].add(e2)
        self.labels = np.zeros((self.queries.shape[0], self.args.num_entities))
        self.filtered_labels = np.zeros(
            (self.queries.shape[0], self.args.num_entities))
        for i, query in enumerate(self.queries):
            e1, r = query[0], query[1]
            e2 = e2_list[i]
            self.labels[i, e2] = 1
            self.filtered_labels[i, list(pos_samples[(e1, r)] - {e2})] = 1

    def __len__(self):
        return self.queries.shape[0]

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx], self.filtered_labels[idx]
