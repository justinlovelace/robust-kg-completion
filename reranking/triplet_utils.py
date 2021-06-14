from triplet_models import TripletTextBert
import os
from TRIPLET_CONSTANTS import DATA_DIR
import argparse
from triplet_dataset import KbTextDataset, KbTextEvalGenerator
import json
import torch


def get_dataset(split: str, args: argparse.ArgumentParser):
    assert split in {'train', 'valid', 'test'}

    if split == 'train':
        return KbTextDataset(args)
    else:
        return KbTextEvalGenerator(split, args)


def get_kg_dicts(args):
    entity_file = 'entity_idx.json'
    unique_entities = json.load(
        open(os.path.join(args.dataset_folder, entity_file)))
    # print(unique_entities)

    entity_file = 'entity_names.json'
    unique_entity_names = json.load(
        open(os.path.join(args.dataset_folder, entity_file)))
    # print(unique_entity_names)

    relation_file = 'rel_idx.json'
    unique_relations = json.load(
        open(os.path.join(args.dataset_folder, relation_file)))
    # print(unique_relations)
    return unique_entities, unique_entity_names, unique_relations


def set_kg_stats(args):
    entity_file = 'entity_idx.json'
    unique_entities = json.load(
        open(os.path.join(args.dataset_folder, entity_file)))
    args.num_entities = len(unique_entities)

    relation_file = 'rel_idx.json'
    unique_relations = json.load(
        open(os.path.join(args.dataset_folder, relation_file)))
    args.num_relations = len(unique_relations)
    print(f'{args.num_entities} unique entities and {args.num_relations} unique relations')


def get_model(args):
    return TripletTextBert(args)

def load_args(args):
    assert len(args.args_dir) > 0
    config = json.load(open(os.path.join(args.args_dir, 'args.json')))
    for key in config:
        if key == 'num_epochs' or key=='grid_search':
            continue
        args.__dict__[key] = config[key]

def get_optimizer(model, args):
    params, heldout_params = [], []
    no_decay = ['prelu', 'bn', 'bias', 'lin_comb', 'scalar'] 
    for name, p in model.named_parameters():
        if p.requires_grad == False or any(nd in name for nd in no_decay):
            heldout_params += [p]
        else:
            # print(name)
            params += [p]
    optimizer = torch.optim.AdamW(
        [
            {'params': params, 'weight_decay': args.weight_decay},
            {'params': heldout_params, 'weight_decay': 0},
        ],
        lr=args.lr)
    return optimizer

def set_model_dir(args):
    model_dir = f'{args.dataset}/top{args.topk}/{args.model}/'
    if args.teacher_lambda>0:
        model_dir = f'{args.dataset}/top{args.topk}/{args.model}/kd'
    else:
        model_dir = f'{args.dataset}/top{args.topk}/{args.model}/no_kd'
    args.output_dir = os.path.abspath(os.path.join(args.save_dir, model_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f'Will save model to {args.output_dir}')

def unroll_tokenizer_dicts(triplets):
    assert len(triplets) > 0
    flattened_triplets = {}
    for k in triplets[0].keys():
        flattened_triplets[k] = torch.cat([d[k] for d in triplets]).to('cuda', non_blocking=True)
    return flattened_triplets

