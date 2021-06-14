import json
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
import sys
from CONSTANTS import DATA_DIR
import os
import math

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


class DistMult(nn.Module):

    def __init__(self, args):
        super(DistMult, self).__init__()
        self.ent_embedding = torch.nn.Embedding(
            args.num_entities, args.embedding_dim)
        self.rel_embedding = torch.nn.Embedding(
            args.num_relations, args.embedding_dim)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.init()

    def init(self):
        xavier_normal_(self.rel_embedding.weight.data)
        xavier_normal_(self.ent_embedding.weight.data)

    def query_emb(self, e1, rel):
        e1_embedded = self.ent_embedding(e1)
        e1_embedded = self.inp_drop(e1_embedded)

        rel_embedded = self.rel_embedding(rel)
        rel_embedded = self.inp_drop(rel_embedded)

        return e1_embedded.mul(rel_embedded)

    def forward(self, queries):
        e1 = queries[:, 0]
        rel = queries[:, 1]

        x = self.query_emb(e1, rel)

        scores = torch.mm(x, self.ent_embedding.weight.t())
        return scores


class Complex(torch.nn.Module):
    def __init__(self, args):
        super(Complex, self).__init__()
        self.emb_e_real = torch.nn.Embedding(
            args.num_entities, args.embedding_dim)
        self.emb_e_img = torch.nn.Embedding(
            args.num_entities, args.embedding_dim)
        self.emb_rel_real = torch.nn.Embedding(
            args.num_entities, args.embedding_dim)
        self.emb_rel_img = torch.nn.Embedding(
            args.num_entities, args.embedding_dim)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.init()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def query_emb(self, e1, rel):
        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img = self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        return e1_embedded_real*rel_embedded_real, e1_embedded_real*rel_embedded_img, e1_embedded_img*rel_embedded_real, e1_embedded_img*rel_embedded_img

    def forward(self, queries):
        e1 = queries[:, 0]
        rel = queries[:, 1]

        realreal, realimg, imgreal, imgimg = self.query_emb(e1, rel)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(
            realreal, self.emb_e_real.weight.transpose(1, 0))
        realimgimg = torch.mm(realimg, self.emb_e_img.weight.transpose(1, 0))
        imgrealimg = torch.mm(imgreal, self.emb_e_img.weight.transpose(1, 0))
        imgimgreal = torch.mm(imgimg, self.emb_e_real.weight.transpose(1, 0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal

        return pred


class ConvE(torch.nn.Module):
    def __init__(self, args):
        super(ConvE, self).__init__()
        if 'Bert' in args.model:
            self.prelu = torch.nn.PReLU(args.embedding_dim)
            self.emb_dim1 = 12
            self.emb_dim2 = args.embedding_dim // self.emb_dim1
        else:
            self.emb_dim1 = 10
            self.emb_dim2 = math.ceil(args.embedding_dim / self.emb_dim1)
            args.embedding_dim = self.emb_dim1 * self.emb_dim2

        self.ent_embedding = torch.nn.Embedding(
            args.num_entities, args.embedding_dim)
        self.rel_embedding = torch.nn.Embedding(
            args.num_relations, args.embedding_dim)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(args.feature_map_dropout)

        self.conv1 = torch.nn.Conv2d(1, args.channels, (3, 3), 1, 0)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.channels)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(args.num_entities)))

        assert self.emb_dim1 * self.emb_dim2 == args.embedding_dim

        # Total number of features produced by the convolution
        fc_dim = (self.emb_dim1*2-2)*(self.emb_dim2-2)*args.channels
        self.fc = torch.nn.Linear(fc_dim, args.embedding_dim)

        self.init()

        self.args = args

    def init(self):
        xavier_normal_(self.ent_embedding.weight.data)
        xavier_normal_(self.rel_embedding.weight.data)

    def query_emb(self, e1, rel):
        batch_size = e1.shape[0]

        e1_embedded = self.ent_embedding(
            e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.rel_embedding(
            rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.feature_map_drop(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if 'Bert' in self.args.model:
            x = self.prelu(x)
        else:
            x = self.bn2(x)
            x = F.relu(x)
        return x

    def forward(self, queries):
        e1 = queries[:, 0]
        rel = queries[:, 1]

        x = self.query_emb(e1, rel)

        scores = torch.mm(x, self.ent_embedding.weight.t())
        scores += self.b.expand_as(scores)
        return scores

    def compute_scores(self, queries):
        e1 = queries[:, 0]
        rel = queries[:, 1]

        x = self.query_emb(e1, rel)

        scores = torch.mm(x, self.ent_embedding.weight.t())
        scores += self.b.expand_as(scores)
        return scores


class ConvTransE(nn.Module):
    def __init__(self, args):
        super(ConvTransE, self).__init__()
        self.ent_embedding = torch.nn.Embedding(
            args.num_entities, args.embedding_dim)
        self.rel_embedding = torch.nn.Embedding(
            args.num_relations, args.embedding_dim)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(args.feature_map_dropout)

        kernel_size = 5
        self.channels = args.channels

        self.conv1 = nn.Conv1d(2, self.channels, kernel_size,
                               stride=1, padding=int(math.floor(kernel_size/2)))

        self.register_parameter('b', Parameter(torch.zeros(args.num_entities)))

        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(self.channels)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.fc = torch.nn.Linear(
            self.channels * args.embedding_dim, args.embedding_dim)
        self.init()

        if 'Bert' in args.model:
            self.prelu = torch.nn.PReLU(args.embedding_dim)

        self.args = args

    def init(self):
        xavier_normal_(self.ent_embedding.weight.data)
        xavier_normal_(self.rel_embedding.weight.data)

    def query_emb(self, e1, rel):
        batch_size = e1.shape[0]

        e1_embedded = self.ent_embedding(e1)
        rel_embedded = self.rel_embedding(rel)

        e1_embedded = e1_embedded.unsqueeze(1)
        rel_embedded = rel_embedded.unsqueeze(1)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)

        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if 'Bert' in self.args.model:
            x = self.prelu(x)
        else:
            x = self.bn2(x)
            x = F.relu(x)
        return x

    def forward(self, queries):
        e1 = queries[:, 0]
        rel = queries[:, 1]

        x = self.query_emb(e1, rel)

        scores = torch.mm(x, self.ent_embedding.weight.t())
        scores += self.b.expand_as(scores)
        return scores

    def compute_scores(self, queries):
        e1 = queries[:, 0]
        rel = queries[:, 1]

        x = self.query_emb(e1, rel)

        scores = torch.mm(x, self.ent_embedding.weight.t())
        scores += self.b.expand_as(scores)
        return scores


class PretrainedBertConvTransE(nn.Module):
    def __init__(self, args):
        super(PretrainedBertConvTransE, self).__init__()
        file_path = os.path.join(
            DATA_DIR[args.dataset], f'{args.bert_model}_{args.bert_pool}.pt')
        embedding = torch.load(file_path)
        args.embedding_dim = embedding.shape[1]
        self.model = ConvTransE(args)
        self.model.ent_embedding = nn.Embedding.from_pretrained(embedding)

    def forward(self, queries):
        return self.model.forward(queries)

    def compute_scores(self, queries):
        return self.model.compute_scores(queries)


class PretrainedBertConvE(nn.Module):
    def __init__(self, args):
        super(PretrainedBertConvE, self).__init__()
        file_path = os.path.join(
            DATA_DIR[args.dataset], f'{args.bert_model}_{args.bert_pool}.pt')
        embedding = torch.load(file_path)
        args.embedding_dim = embedding.shape[1]
        self.model = ConvE(args)
        self.model.ent_embedding = nn.Embedding.from_pretrained(embedding)

    def forward(self, queries):
        return self.model.forward(queries)

    def compute_scores(self, queries):
        return self.model.compute_scores(queries)


class ConvBlock(nn.Module):
    def __init__(self, args, input_dim, output_dim, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, (3, 3), padding=1, padding_mode='circular')

        self.feature_map_drop = torch.nn.Dropout2d(args.feature_map_dropout)

        self.bn = torch.nn.BatchNorm2d(input_dim)

    def forward(self, features):
        x = self.bn(features)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = self.conv(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, args, input_dim, int_dim, output_dim, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, int_dim, (1, 1), stride=stride)
        self.conv2 = nn.Conv2d(
            int_dim, int_dim, (3, 3), padding=1, padding_mode='circular')
        self.conv3 = nn.Conv2d(int_dim, output_dim, (1, 1))

        self.feature_map_drop = torch.nn.Dropout2d(args.feature_map_dropout)

        self.bn1 = torch.nn.BatchNorm2d(input_dim)
        self.bn2 = torch.nn.BatchNorm2d(int_dim)
        self.bn3 = torch.nn.BatchNorm2d(int_dim)

        if input_dim != output_dim:
            self.proj_shortcut = nn.Conv2d(
                input_dim, output_dim, (1, 1), stride=stride)
        else:
            self.proj_shortcut = None

    def init(self):
        xavier_normal_(self.rel_embedding.weight.data)

    def forward(self, features):
        x = self.bn1(features)
        x = F.relu(x)
        if self.proj_shortcut:
            features = self.proj_shortcut(features)
        x = self.feature_map_drop(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = self.conv3(x)

        return x + features


class PretrainedBertResNet(nn.Module):
    def __init__(self, args):
        super(PretrainedBertResNet, self).__init__()
        args.embedding_dim = 768

        file_path = os.path.join(
            DATA_DIR[args.dataset], 'bert_emb.pt')
        embedding = torch.load(file_path)
        args.embedding_dim = embedding.shape[1]
        self.ent_embedding = nn.Embedding.from_pretrained(embedding)

        self.rel_embedding = torch.nn.Embedding(
            args.num_relations, args.embedding_dim)

        self.reshape_len = args.reshape_len

        self.conv1 = nn.Conv1d(2, self.reshape_len**2, kernel_size=1)

        bottlenecks = []

        input_dim = args.embedding_dim
        output_dim = args.embedding_dim
        for i in range(args.resnet_num_blocks):
            if i == 0:
                bottlenecks.extend([BottleneckBlock(
                    args, input_dim, output_dim//4, output_dim) for _ in range(args.resnet_block_depth)])
            else:
                bottlenecks.append(BottleneckBlock(
                    args, input_dim, output_dim//4, output_dim, stride=1))
                input_dim = output_dim
                bottlenecks.extend([BottleneckBlock(
                    args, input_dim, output_dim//4, output_dim) for _ in range(min(args.resnet_block_depth, 2)-1)])
            output_dim *= 2
        self.output_dim = output_dim//2
        self.bottlenecks = nn.Sequential(*bottlenecks)

        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(args.feature_map_dropout)

        self.register_parameter('b', Parameter(torch.zeros(args.num_entities)))

        self.fc = nn.Linear(self.output_dim, args.embedding_dim)

        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm2d(self.output_dim)
        self.bn2 = torch.nn.BatchNorm2d(args.embedding_dim)
        self.bn3 = torch.nn.BatchNorm2d(args.embedding_dim)
        self.args = args
        self.init()
        self.prelu = torch.nn.PReLU(args.embedding_dim)

    def init(self):
        xavier_normal_(self.rel_embedding.weight)

    def query_emb(self, e1, rel):
        batch_size = e1.shape[0]

        ent_embedded = self.ent_embedding(e1).unsqueeze(1)
        rel_embedded = self.rel_embedding(rel).unsqueeze(1)

        stacked_inputs = torch.cat(
            [ent_embedded, rel_embedded], 1)
        x = self.bn0(stacked_inputs)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = torch.transpose(x, 1, 2).view(
            batch_size, self.args.embedding_dim, self.reshape_len, self.reshape_len).contiguous()

        x = self.bottlenecks(x)

        x = self.bn1(x)
        x = F.relu(x)

        x = torch.mean(x.view(batch_size, self.output_dim, -1), dim=2)
        x = self.hidden_drop(x)
        x = self.fc(x)

        x = self.prelu(x)
        x = self.hidden_drop(x)

        return x

    def forward(self, queries):
        e1 = queries[:, 0]
        rel = queries[:, 1]

        x = self.query_emb(e1, rel)

        scores = torch.mm(x, self.ent_embedding.weight.t())
        scores += self.b.expand_as(scores)
        return scores

    def compute_scores(self, queries):
        e1 = queries[:, 0]
        rel = queries[:, 1]

        x = self.query_emb(e1, rel)

        if self.args.bert_model == 'none' or self.args.bert_layer_comb == 'none':
            scores = torch.mm(x, self.ent_embedding.weight.t())
        else:
            scores = torch.mm(x, self.ent_embedding.get_transposed_weight())
        scores += self.b.expand_as(scores)
        return scores


class PretrainedBertDeepConv(nn.Module):
    def __init__(self, args):
        super(PretrainedBertDeepConv, self).__init__()

        file_path = os.path.join(
            DATA_DIR[args.dataset], 'bert_emb.pt')
        embedding = torch.load(file_path)
        args.embedding_dim = embedding.shape[1]
        self.ent_embedding = nn.Embedding.from_pretrained(embedding)

        self.rel_embedding = torch.nn.Embedding(
            args.num_relations, args.embedding_dim)

        self.reshape_len = args.reshape_len

        self.conv1 = nn.Conv1d(2, self.reshape_len**2, kernel_size=1)

        conv_blocks = []

        # input_dim = args.embedding_dim
        # output_dim = args.embedding_dim
        conv_blocks = [ConvBlock(args, args.embedding_dim, args.embedding_dim),
                       ConvBlock(args, args.embedding_dim,
                                 2*args.embedding_dim),
                       ConvBlock(args, 2*args.embedding_dim,
                                 2*args.embedding_dim),
                       ]
        self.output_dim = args.embedding_dim*2
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # self.conv_final = nn.Conv2d(self.output_dim, 768, (1,1))

        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(args.feature_map_dropout)

        self.register_parameter('b', Parameter(torch.zeros(args.num_entities)))

        self.fc = nn.Linear(self.output_dim, args.embedding_dim)
        # self.fc = nn.Linear(self.output_dim*(self.reshape_len**2), args.embedding_dim)

        # self.bn0 = torch.nn.GroupNorm(2, 2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        if args.gn:
            self.bn1 = torch.nn.GroupNorm(32, self.output_dim)
            self.bn2 = torch.nn.GroupNorm(32, args.embedding_dim)
            self.bn3 = torch.nn.GroupNorm(32, args.embedding_dim)
        else:
            self.bn1 = torch.nn.BatchNorm2d(self.output_dim)
            self.bn2 = torch.nn.BatchNorm2d(args.embedding_dim)
            self.bn3 = torch.nn.BatchNorm2d(args.embedding_dim)
        self.args = args
        self.init()
        self.prelu = torch.nn.PReLU(args.embedding_dim)

    def init(self):
        if self.args.bert_model == 'none':
            xavier_normal_(self.ent_embedding.weight)
        xavier_normal_(self.rel_embedding.weight)

    def query_emb(self, e1, rel):
        batch_size = e1.shape[0]

        ent_embedded = self.ent_embedding(e1).unsqueeze(1)
        rel_embedded = self.rel_embedding(rel).unsqueeze(1)

        stacked_inputs = torch.cat(
            [ent_embedded, rel_embedded], 1)
        x = self.bn0(stacked_inputs)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = torch.transpose(x, 1, 2).view(
            batch_size, self.args.embedding_dim, self.reshape_len, self.reshape_len).contiguous()

        x = self.conv_blocks(x)

        x = self.bn1(x)
        x = F.relu(x)
        # x = self.conv_final(x)
        # x = self.bn2(x)
        # x = F.relu(x)
        # x = torch.mean(x.view(batch_size, 768, -1), dim=2)
        # x = self.hidden_drop(x)

        x = torch.mean(x.view(batch_size, self.output_dim, -1), dim=2)
        x = self.hidden_drop(x)
        x = self.fc(x)
        # x = x.view(batch_size, -1)
        # x = self.hidden_drop(x)
        # x = self.fc(x)

        x = self.prelu(x)
        x = self.hidden_drop(x)

        return x

    def forward(self, queries):
        e1 = queries[:, 0]
        rel = queries[:, 1]

        x = self.query_emb(e1, rel)

        if self.args.bert_model == 'none' or self.args.bert_layer_comb == 'none':
            scores = torch.mm(x, self.ent_embedding.weight.t())
        else:
            scores = torch.mm(x, self.ent_embedding.get_transposed_weight())
        scores += self.b.expand_as(scores)
        return scores

    def compute_scores(self, queries):
        e1 = queries[:, 0]
        rel = queries[:, 1]

        x = self.query_emb(e1, rel)

        if self.args.bert_model == 'none' or self.args.bert_layer_comb == 'none':
            scores = torch.mm(x, self.ent_embedding.weight.t())
        else:
            scores = torch.mm(x, self.ent_embedding.get_transposed_weight())
        scores += self.b.expand_as(scores)
        return scores
