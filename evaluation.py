import numpy as np
import os
import argparse
from dataset import KbEvalGenerator
import json
import torch
from tqdm import tqdm
import sys
import utils
import pandas as pd
from torch.utils.data import DataLoader
from CONSTANTS import DATA_DIR

def ranking_and_hits(model, eval_loader, args, write_ranks=False):
    model.eval()

    ranks = []

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            queries, labels, filtered_labels = batch[0].to('cuda', non_blocking=True), batch[1].to('cuda', non_blocking=True), batch[2].to('cuda', non_blocking=True)
            preds = torch.sigmoid(model(queries))
            # Filter other positive triples to bottom of ranking
            preds[torch.nonzero(filtered_labels, as_tuple=True)] = -1
                
            batch_ranks = torch.argsort(torch.argsort(preds, dim=1, descending=True), dim=1) + 1

            ranks.append(batch_ranks[torch.nonzero(labels, as_tuple=True)].to('cpu'))           


    ranks = np.array(torch.cat(ranks, dim=0).type(torch.FloatTensor))

    if write_ranks:
        torch.save(ranks, os.path.join(args.output_dir, 'ranks.np'))

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


def main(args):
    eval_dataset = utils.get_dataset('test', args)     
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, pin_memory=True, shuffle=False)
    model = utils.get_model(args)
    checkpoint = torch.load(os.path.join(args.output_dir, 'state_dict.pt'), map_location='cpu')

    model.eval()
    with torch.no_grad():
        model.load_state_dict(checkpoint['state_dict'])

    # Use GPU
    if torch.cuda.is_available():
        model.to('cuda')
    else:
        raise RuntimeError

    print("===========TEST============")
    metrics = ranking_and_hits(
        model, eval_loader, args, write_ranks=True)
    print(json.dumps(metrics))
    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
        f.write(json.dumps(metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Options for Knowledge Base Completion')

    # General
    parser.add_argument("--model", type=str, required=False, default='PretrainedBertResNet',
                        help="model to use")
    parser.add_argument("--dataset", type=str, required=False, default='CN100K',
                        help="dataset to use")
    parser.add_argument("--save_dir", type=str, required=False, default="saved_models",
                        help="directory where the models are saved")
    args = parser.parse_args()

    args.dataset_folder = DATA_DIR[args.dataset]
    utils.set_kg_stats(args)

    model_dir = f'{args.save_dir}/{args.dataset}/{args.model}/'
    print(model_dir)

    with open(os.path.join(model_dir, 'args.json'), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    args.write_results = True
    args.output_dir = model_dir
    main(args)
