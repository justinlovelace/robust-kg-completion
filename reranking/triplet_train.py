import torch
import argparse
import triplet_utils
import random
import numpy as np
import json
import typing
import os
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
from triplet_evaluation import ranking_and_hits
from TRIPLET_CONSTANTS import DATA_DIR, BERT_IDS
# import contextlib
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(model, trainloader, optimizer, loss_fn, args, epoch, writer=None, eval_dataset=None, eval_loader=None):
    model.train()
    running_loss = 0.0
    batches = tqdm(trainloader)

    for i, data in enumerate(batches):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = {key: val.to('cuda', non_blocking=False) for key, val in data[0].items(
        )}, data[1].to('cuda', non_blocking=False)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        if args.clip != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # print('stepping optimizer...')
        optimizer.step()
        
        
        if i % 250 == 249:    # update every 250 mini-batches
            batches.set_postfix(loss=running_loss / i)        
    
    return running_loss/(i+1)


def main(args: argparse.ArgumentParser, writer=None):
    print('Setting seeds for reproducibility...')
    set_seeds(args.seed)
    train_dataset = triplet_utils.get_dataset('train', args)
    trainloader = DataLoader(
        train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers, drop_last=True)
    model = triplet_utils.get_model(args)

    eval_dataset = triplet_utils.get_dataset('valid', args)
    with open(os.path.join(args.output_dir, 'baseline_metrics.json'), 'w') as f:
        f.write(json.dumps(eval_dataset.baseline_metrics))
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers)
    

    # Use GPU
    if torch.cuda.is_available():
        model.to('cuda')
    else:
        raise RuntimeError

    # optimizer
    optimizer = triplet_utils.get_optimizer(model, args)


    # loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_mrr = 0
    epochs_since_improvement = 0

    for epoch in range(args.num_epochs):
        train_loss = train(model, trainloader, optimizer, loss_fn, args, epoch, writer, eval_dataset, eval_loader)

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
                        os.path.join(args.output_dir, 'state_dict.pt'), _use_new_zipfile_serialization=False)
            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                f.write(json.dumps(metrics))

        # print(model.bert.embeddings.position_ids)
        print("Epoch {:04d} | Best MRR {:.5f} | Current MRR {:.5f} | Current MR {:.5f} | H@1 {:.5f} | H@3 {:.5f} | H@5 {:.5f} | H@10 {:.5f} | Epochs Since Improvement {:04d}".
              format(epoch, best_mrr, mrr, metrics['MR'], metrics['H@1'], metrics['H@3'], metrics['H@5'], metrics['H@10'], epochs_since_improvement))
        print("Baseline Metrics | MRR {:.5f} | MR {:.5f} | H@1 {:.5f} | H@3 {:.5f} | H@5 {:.5f} | H@10 {:.5f}".
              format(eval_dataset.baseline_metrics['MRR'], eval_dataset.baseline_metrics['MR'], eval_dataset.baseline_metrics['H@1'], eval_dataset.baseline_metrics['H@3'], eval_dataset.baseline_metrics['H@5'], eval_dataset.baseline_metrics['H@10']))
    print("Epoch {:04d} | Best MRR {:.5f} | Current MRR {:.5f} | Current MR {:.5f} | H@1 {:.5f} | H@3 {:.5f} | H@5 {:.5f} | H@10 {:.5f} | Epochs Since Improvement {:04d}".
            format(epoch, best_mrr, mrr, metrics['MR'], metrics['H@1'], metrics['H@3'], metrics['H@5'], metrics['H@10'], epochs_since_improvement))
    print("Baseline Metrics | MRR {:.5f} | MR {:.5f} | H@1 {:.5f} | H@3 {:.5f} | H@5 {:.5f} | H@10 {:.5f}".
            format(eval_dataset.baseline_metrics['MRR'], eval_dataset.baseline_metrics['MR'], eval_dataset.baseline_metrics['H@1'], eval_dataset.baseline_metrics['H@3'], eval_dataset.baseline_metrics['H@5'], eval_dataset.baseline_metrics['H@10']))
    if writer:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Options for Knowledge Base Completion')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # General
    parser.add_argument("--model", type=str, required=False, default='DistMult',
                        help="model to use")
    parser.add_argument("--dataset", type=str, required=False, default='UMLS1',
                        help="dataset to use")
    parser.add_argument("--teacher_folder", type=str, required=False, default='',
                        help="folder of ranking model")
    parser.add_argument("--topk", type=int, required=True, default=10,
                        help="folder of ranking model")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="number of maximum training epochs")
    parser.add_argument("--patience", type=int, default=3,
                        help="number of minimum training epochs")
    parser.add_argument("--save_dir", type=str, required=False, default="saved_reranking_models",
                        help="output directory to store metrics and model file")
    parser.add_argument("--eval_batch_size", type=int, default=512,
                        help="batch size when evaluating")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed value")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="use fewer eval instances in debugging mode")


    # Miscellaneous Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay coefficient")
    parser.add_argument("--clip", type=int, default=0,
                        help="value used for gradient clipping")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--teacher_lambda", type=float, default=0.5,
                        help="Whether to set hyperparameters from experiment")
    parser.add_argument("--temperature", type=float, default=1.,
                        help="Scales logits")      
    args = parser.parse_args()

    args.eval_batch_size = args.eval_batch_size//args.topk

    torch.set_num_threads(args.num_workers)

    args.dataset_folder = DATA_DIR[args.dataset]
    triplet_utils.set_kg_stats(args)
    args.bert_model = BERT_IDS[args.dataset]


    try:   
        triplet_utils.set_model_dir(args)
        writer = SummaryWriter(args.output_dir)
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        main(args, writer)
    except KeyboardInterrupt:
        os.rename(args.output_dir, args.output_dir+'terminated')
        print('Interrupted')
