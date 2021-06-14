import sys
sys.path.append(".")

import pandas as pd
import json
import os
from CONSTANTS import DATA_DIR

def extract_csv(dataset):
    file_path = os.path.join(DATA_DIR[dataset], 'entity_idx.json')
    e2idx = json.load(open(file_path))

    file_path = os.path.join(DATA_DIR[dataset], 'rel_idx.json')
    rel2idx = json.load(open(file_path))

    splits = ['train', 'valid', 'test']
    filenames = ['train.txt', 'valid.txt', 'test.txt']
    for split in splits:
        # Hack to order the data a certain way for FB15K-237-Sparse for exact reproduction
        if 'SPARSE' in dataset and split == 'train':
            data = []
            data_rev = []
        else:
            data = []
        with open(os.path.join(DATA_DIR[dataset], f'{split}.txt')) as f:
            for i, line in enumerate(f):
                if 'FB15K' in dataset:
                    e1, rel, e2 = line.split('\t')
                else:
                    rel, e1, e2 = line.split('\t')
                e1 = e1.strip()
                e2 = e2.strip()
                rel = rel.strip()
                rel_reverse = rel+ '_reverse'
                data.append([e1, e2idx[e1], rel, rel2idx[rel], e2, e2idx[e2]])
                if 'SPARSE' in dataset and split == 'train': 
                    data_rev.append([e2, e2idx[e2], rel_reverse, rel2idx[rel_reverse], e1, e2idx[e1]])
                else: 
                    data.append([e2, e2idx[e2], rel_reverse, rel2idx[rel_reverse], e1, e2idx[e1]])
            if 'SPARSE' in dataset and split == 'train': 
                data = data + data_rev
        df = pd.DataFrame(data, columns = ['entity1', 'entity1_id', 'rel', 'rel_id', 'entity2', 'entity2_id'])  
        file_path = os.path.join(DATA_DIR[dataset], f'df_{split}.csv')
        print(f'Saving {file_path}')
        df.to_csv(file_path)


def main():
    for dataset in ['FB15K_237', 'CN100K', 'FB15K_237_SPARSE']:
        print(f'Processing {dataset}...')
        extract_csv(dataset)
    print('done')

if __name__ == '__main__':
    main()