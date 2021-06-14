import torch
import argparse
import utils
import random
import numpy as np
import json
import typing
import os
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
from evaluation import ranking_and_hits
from CONSTANTS import DATA_DIR, SHARED_PARAMS, MODEL_PARAMS, GRID_SEARCH
import time
from functools import partial
import contextlib
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(model, trainloader, optimizer, loss_fn, args, epoch):
    model.train()
    running_loss = 0.0
    batches = tqdm(trainloader)
    num_batches = len(trainloader)

    for i, data in enumerate(batches):
        inputs, labels = data[0].to('cuda', non_blocking=False), data[1].to(
            'cuda', non_blocking=False)

        if 'to_n' in args.strategy and 'mask' not in args.strategy:
            labels = (1.0 - args.label_smoothing_epsilon)*labels + (args.label_smoothing_epsilon/args.num_entities)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        if args.clip != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        
        if i % 50 == 49:
            batches.set_postfix(loss=running_loss / i)
        
    return running_loss/(i+1)


def main(args: argparse.ArgumentParser, writer=None):
    print('Setting seeds for reproducibility...')
    set_seeds(args.seed)
    train_dataset = utils.get_dataset('train', args)
    trainloader = DataLoader(
    train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers, drop_last=True)
    

    eval_dataset = utils.get_dataset('valid', args)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers)
    model = utils.get_model(args)

    # Use GPU
    if torch.cuda.is_available():
        model.to('cuda')
    else:
        raise RuntimeError

    # optimizer
    optimizer = utils.get_optimizer(model, args)

    # loss function
    if 'softmax' in args.strategy:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    best_mrr = 0
    epochs_since_improvement = 0

    for epoch in range(args.num_epochs):
        # print(f'Epoch {epoch}')
        train_loss = train(model, trainloader, optimizer, loss_fn, args, epoch)

        metrics = ranking_and_hits(
            model, eval_loader, args)
        mrr = metrics['MRR']
        

        if writer:
            writer.add_scalar('MRR/val', mrr, epoch)
            writer.add_scalar('train/loss', train_loss, epoch)

        if mrr < best_mrr:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                break
        else:
            epochs_since_improvement = 0
            best_mrr = mrr
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                        os.path.join(args.output_dir, 'state_dict.pt'))
            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                f.write(json.dumps(metrics))

        
        print("Epoch {:04d} | Best MRR {:.5f} | Current MRR {:.5f} | H@1 {:.5f} | H@10 {:.5f} | Epochs Since Improvement {:04d}".
              format(epoch, best_mrr, mrr, metrics['H@1'], metrics['H@10'], epochs_since_improvement))
    print("Epoch {:04d} | Best MRR {:.5f} | Current MRR {:.5f} | H@1 {:.5f} | H@10 {:.5f} | Epochs Since Improvement {:04d}".
              format(epoch, best_mrr, mrr, metrics['H@1'], metrics['H@10'], epochs_since_improvement))
    if writer:
        writer.close()            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Options for Knowledge Base Completion')

    # General
    parser.add_argument("--model", type=str, required=False, default='PretrainedBertResNet',
                        help="model to use")
    parser.add_argument("--dataset", type=str, required=False, default='CN100K',
                        help="dataset to use")
    parser.add_argument("--strategy", type=str, required=False, default='one_to_n',
                        help="training strategy")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="number of maximum training epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="patience for early stopping")
    parser.add_argument("--save_dir", type=str, required=False, default="saved_models",
                        help="output directory to store metrics and model file")
    parser.add_argument("--eval_batch_size", type=int, default=128,
                        help="batch size when evaluating")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed value")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of threads")


    # Model Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--input_dropout", type=float, default=0,
                        help="input dropout")
    parser.add_argument("--feature_map_dropout", type=float, default=0,
                        help="feature map dropout")
    parser.add_argument("--label_smoothing_epsilon", type=float, default=0.1,
                        help="epsilon for label smoothing")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay coefficient")
    parser.add_argument("--clip", type=int, default=0,
                        help="value used for gradient clipping")
    parser.add_argument("--embedding_dim", type=int, default=200,
                        help="embedding dimension for entities and relations")
    parser.add_argument("--channels", type=int, default=200,
                    help="output dimension of convolution")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--reshape_len", type=int, default=4,
                        help="Side length for deep convolutional models")
    parser.add_argument("--resnet_num_blocks", type=int, default=2,
                        help="Number of resnet blocks")
    parser.add_argument("--resnet_block_depth", type=int, default=3,
                        help="Depth of each resnet block")
    parser.add_argument("--args_dir", type=str, default='',
                        help="Whether to set hyperparameters from experiment")
    args = parser.parse_args()

    torch.set_num_threads(args.num_workers)

    args.dataset_folder = DATA_DIR[args.dataset]
    utils.set_kg_stats(args)

    if args.args_dir != '':
        utils.load_args(args)

    try:   
        utils.set_model_dir(args)
        writer = SummaryWriter(args.output_dir)
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        main(args, writer)
    except KeyboardInterrupt:
        os.rename(args.output_dir, args.output_dir+'terminated')
        print('Interrupted')    
