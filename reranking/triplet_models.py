import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import transformers
import sys

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

class TripletTextBert(nn.Module):
    def __init__(self, args):
        super(TripletTextBert, self).__init__()
        args.embedding_dim = 768
        self.bert = transformers.BertModel.from_pretrained(args.bert_model)
        # print('resizing')
        self.bert.resize_token_embeddings(args.vocab_size) 
        # print('resizing')
        print(args.vocab_size)
        # The new vector is added at the end of the embedding matrix
        self.fc = nn.Linear(args.embedding_dim, 1)
        self.args = args

        self.lin_comb = nn.Parameter(torch.zeros(
                13, dtype=torch.float32, requires_grad=True))
        self.softmax = torch.nn.Softmax(dim=0)

        self.dropout = torch.nn.Dropout(p=args.dropout)

    def forward(self, encoded):
        x = self.bert(**encoded, output_hidden_states=True)[2]
        scalars = self.softmax(self.lin_comb)
        embedded = torch.stack(
            [sc*x[idx][:,0,:] for (sc, idx) in zip(scalars, range(13))], dim=1)
        comb_embedded = torch.sum(embedded, dim=1)
        x = self.fc(self.dropout(comb_embedded)).squeeze(-1)
        return x
