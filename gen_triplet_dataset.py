import os
import argparse
import json
import torch
from tqdm import tqdm
import utils
import pandas as pd
from torch.utils.data import DataLoader
from CONSTANTS import DATA_DIR, COLUMN_NAMES
from collections import defaultdict 

def gen_topk_dataset(model, args, split, topk):
    model.eval()

    eval_dataset = utils.get_dataset(split, args)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, pin_memory=True, shuffle=False)   

    dataframe_dict = defaultdict(list)
    e1, rel, e2 = COLUMN_NAMES[args.dataset]
    with torch.no_grad():
        count=0
        for batch in tqdm(eval_loader):
            if split == 'train':
                queries, labels = batch[0].to('cuda', non_blocking=True), batch[1].to('cuda', non_blocking=True)
            else:
                queries, labels, filtered_labels = batch[0].to('cuda', non_blocking=True), batch[1].to('cuda', non_blocking=True), batch[2].to('cuda', non_blocking=True)

            logits = model(queries)
            preds = torch.sigmoid(logits)
            if split != 'train':
                preds[torch.nonzero(filtered_labels, as_tuple=True)] = -1

            top_e2_scores, top_e2s = torch.topk(preds, topk)
            if split != 'train':
                topk_labels = torch.gather(labels, dim=1, index=top_e2s)
                # Check if entire batch needs to be filtered out
                if torch.sum(topk_labels).item() == 0:
                    continue
                # Otherwise filter out specific elements in the batch
                indices = torch.nonzero(torch.sum(topk_labels, dim=1), as_tuple=False).squeeze(1)

                top_e2_scores = top_e2_scores[indices]
                top_e2s = top_e2s[indices]
                logits = logits[indices]
                queries = queries[indices]
                labels = labels[indices]
            
            dataframe_dict['scores'].extend(top_e2_scores.tolist())
            
            dataframe_dict['logits'].extend(torch.gather(logits, dim=1, index=top_e2s).tolist())
            dataframe_dict[e1].extend(queries[:, 0].tolist())
            dataframe_dict[rel].extend(queries[:, 1].tolist())
            dataframe_dict[e2].extend(top_e2s.tolist())
            dataframe_dict['labels'].extend(torch.gather(labels, dim=1, index=top_e2s).tolist())
                
    
    df = pd.DataFrame(data=dataframe_dict)
    print(df.head())
    print(df.columns)
    print(len(df))
    save_dir = os.path.join(args.model_dir, 'triplet_datasets')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'df_{split}_{topk}triplets.csv')
    print(f'Saving triplet dataset with {len(df)} triplets to {save_path}')
    df.to_csv(save_path, index=False)
    return    

def main(args):
    args.strategy = 'gen_triplets'
    model = utils.get_model(args)
    checkpoint = torch.load(os.path.join(args.model_dir, 'state_dict.pt'), map_location='cpu')
    model.eval()
    # Use GPU
    if torch.cuda.is_available():
        model.to('cuda')
    else:
        raise RuntimeError

    with torch.no_grad():
        model.load_state_dict(checkpoint['state_dict'])
    for split in ['train', 'valid', 'test']:
        gen_topk_dataset(model, args, split, args.topk)
        # return
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Options for Knowledge Base Completion')

    # General
    parser.add_argument("--model", type=str, required=False, default='DistMult',
                        help="model to use")
    parser.add_argument("--dataset", type=str, required=False, default='UMLS1',
                        help="dataset to use")
    parser.add_argument("--model_dir", type=str, required=False, default='',
                        help="folder storing the pretrained model")
    parser.add_argument("--topk", type=int, required=True, default=10,
                        help="folder storing the pretrained model")
    args = parser.parse_args()

    args.dataset_folder = DATA_DIR[args.dataset]
    utils.set_kg_stats(args)

    with open(os.path.join(args.model_dir, 'args.json'), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    args.write_results = True
    
    main(args)
