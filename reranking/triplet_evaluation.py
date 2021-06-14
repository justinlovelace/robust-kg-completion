import numpy as np
import typing
import os
import argparse
import json
import torch
from tqdm import tqdm
import sys
import pandas as pd
from torch.utils.data import DataLoader
import triplet_utils
from TRIPLET_CONSTANTS import DATA_DIR


def compute_metrics(d, ranks):
    d['MR'] = np.mean(ranks).item()
    d['MRR'] = np.mean(1. / ranks).item()
    d['H@1'] = np.mean(np.where(ranks < 2, 1, 0)).item()
    d['H@3'] = np.mean(np.where(ranks < 4, 1, 0)).item()
    d['H@5'] = np.mean(np.where(ranks < 6, 1, 0)).item()
    d['H@10'] = np.mean(np.where(ranks < 11, 1, 0)).item()

def eval_mix(model, eval_dataset, eval_loader, args, write_results=False):
    model.eval()

    ranks = []
    preds = []
    all_labels = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            triplets, labels = {key: val.to('cuda', non_blocking=True).view(-1, val.shape[-1]) for key, val in batch[0].items()}, batch[1].to('cuda', non_blocking=True)
            
            preds.append(softmax(model(triplets).view(-1, args.topk)).to('cpu'))
            all_labels.append(labels.to('cpu'))

                
        all_preds = torch.cat(preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

    if args.split == 'valid':
        student_lambdas = np.linspace(0,1,101)
        best_mrr = 0        
        for student_lambda in student_lambdas:
            ranks = np.array((torch.argsort(torch.argsort(all_preds*student_lambda+eval_dataset.teacher_labels*(1-student_lambda), dim=1, descending=True), dim=1)[torch.nonzero(all_labels, as_tuple=True)]+ 1).type(torch.FloatTensor)) 

            metrics = {}
            metrics['student_lambda'] = student_lambda
            compute_metrics(metrics, ranks)

            if metrics['MRR'] > best_mrr:
                best_lambda = student_lambda
                best_mrr = metrics['MRR']
                best_metrics=metrics

        print('==============VALIDATION (Top-10)==============')
        print(f'Best lambda: {best_lambda}')
        print(json.dumps(best_metrics)) 
        return best_metrics
    # sys.exit()

    with open(os.path.join(args.output_dir, 'valid_metrics.json')) as f:
            best_val_metrics = json.load(f)
    best_lambda = best_val_metrics['student_lambda']
    print(f'Computing test metrics using lambda: {best_lambda}')
    best_ranks = np.array((torch.argsort(torch.argsort(all_preds*best_lambda+eval_dataset.teacher_labels*(1-best_lambda), dim=1, descending=True), dim=1)[torch.nonzero(all_labels, as_tuple=True)]+ 1).type(torch.FloatTensor)) 
    print('==============TESTING (Top-10)==============')
    metrics = {}
    compute_metrics(metrics, best_ranks)
    print(json.dumps(metrics))

    print('==============Final Metrics (Student Only)==============')
    student_ranks = np.array((torch.argsort(torch.argsort(all_preds, dim=1, descending=True), dim=1)[torch.nonzero(all_labels, as_tuple=True)]+ 1).type(torch.FloatTensor)) 
    teacher_ranks = torch.load(os.path.join(args.teacher_folder, 'ranks.np'))
    assert np.count_nonzero(teacher_ranks < 11) == len(student_ranks)
    print(teacher_ranks.shape)
    teacher_ranks[teacher_ranks < 11] = student_ranks
    print(teacher_ranks.shape)
    student_metrics = {}
    compute_metrics(student_metrics, teacher_ranks)
    print(json.dumps(student_metrics))
    if args.write_results:
        student_output_dir = os.path.abspath(os.path.join(args.output_dir, 'student'))
        if not os.path.exists(student_output_dir):
            os.makedirs(student_output_dir)
        torch.save(teacher_ranks, os.path.join(student_output_dir, 'ranks.np'))
        with open(os.path.join(student_output_dir, f'{args.split}_metrics.json'), 'w') as f:
            f.write(json.dumps(student_metrics))
    
    print('==============Final Metrics==============')
    teacher_ranks = torch.load(os.path.join(args.teacher_folder, 'ranks.np'))
    assert np.count_nonzero(teacher_ranks < 11) == len(best_ranks)
    print(teacher_ranks.shape)
    teacher_ranks[teacher_ranks < 11] = best_ranks
    print(teacher_ranks.shape)
    final_metrics = {}
    compute_metrics(final_metrics, teacher_ranks)
    print(json.dumps(final_metrics))
    if write_results:
        torch.save(teacher_ranks, os.path.join(args.output_dir, 'ranks.np'))
        
    return final_metrics


def ranking_and_hits(model, eval_loader, args):
    model.eval()

    ranks = []

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            if 'Text' in args.model:
                triplets, labels = {key: val.to('cuda', non_blocking=True).view(-1, val.shape[-1]) for key, val in batch[0].items()}, batch[1].to('cuda', non_blocking=True)
            else:
                triplets, labels = batch[0].to('cuda', non_blocking=True).view(-1, 3), batch[1].to('cuda', non_blocking=True)
            preds = torch.sigmoid(model(triplets).view(-1, args.topk))                
            batch_ranks = torch.argsort(torch.argsort(preds, dim=1, descending=True), dim=1) + 1

            ranks.append(batch_ranks[torch.nonzero(labels, as_tuple=True)].to('cpu'))           


    ranks = np.array(torch.cat(ranks, dim=0).type(torch.FloatTensor))
    print(ranks)


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
    eval_dataset = triplet_utils.get_dataset(args.split, args)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=1, drop_last=False)
    model = triplet_utils.get_model(args)
    # Use GPU
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'state_dict.pt'), map_location='cpu')

    with torch.no_grad():
        model.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available():
        model.to('cuda')
    else:
        raise RuntimeError
    model.eval()

    

    if args.action == 'eval_test':
        metrics = ranking_and_hits(
            model, eval_loader, args)
        print(json.dumps(metrics))
        with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
            f.write(json.dumps(metrics))
    elif args.action == 'eval_mix':
        metrics = eval_mix(
            model, eval_dataset, eval_loader, args, write_results=True)
        with open(os.path.join(args.output_dir, f'{args.split}_metrics.json'), 'w') as f:
            f.write(json.dumps(metrics))
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Options for Knowledge Base Completion')

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # General
    parser.add_argument("--model", type=str, required=False, default='TripletTextBert',
                        help="model to use")
    parser.add_argument("--dataset", type=str, required=False, default='SNOMED_CT_CORE',
                        help="dataset to use")
    parser.add_argument("--save_dir", type=str, required=False, default="saved_reranking_models",
                        help="directory where the models are saved")
    parser.add_argument("--action", type=str, required=False, default="eval_test",
                        help="directory where the models are saved")
    parser.add_argument("--topk", type=int, required=False, default=10,
                        help="folder of ranking model")
    parser.add_argument("--split", type=str, required=False, default='test',
                        help="folder of ranking model")
    parser.add_argument("--teacher_lambda", type=float, default=0.5,
                        help="Whether to set hyperparameters from experiment")
    args = parser.parse_args()

    args.dataset_folder = DATA_DIR[args.dataset]
    triplet_utils.set_kg_stats(args)
    triplet_utils.set_model_dir(args)

    with open(os.path.join(args.output_dir, 'args.json'), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    args.write_results = True
    args.eval_batch_size = 512//args.topk
    
    main(args)

