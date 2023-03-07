import torch
import torch.nn.functional as F
import dgl
import numpy as np
import random
from dgl.readout import sum_nodes
from dgl.nn.pytorch.conv import RelGraphConv
from dgl.nn.pytorch.conv import GraphConv
from torch import nn
import pandas as pd
from functools import reduce


class ResGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=F.relu, residual=True, batchnorm=True):
        super(ResGCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv_layer = GraphConv(in_feats, out_feats, bias=True, activation=activation)
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, node_feats):
        
        new_feats = self.graph_conv_layer(bg, node_feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        del res_feats
        torch.cuda.empty_cache()
        return new_feats

class WeightAndSum(nn.Module):
    def __init__(self, in_feats, task_num=1, attention=True, return_weight=False):
        super(WeightAndSum, self).__init__()
        self.attention = attention
        self.in_feats = in_feats
        self.task_num = task_num
        self.return_weight=return_weight
        self.atom_weighting_specific = nn.ModuleList([self.atom_weight(self.in_feats) for _ in range(self.task_num)])
        self.shared_weighting = self.atom_weight(self.in_feats)
    def forward(self, bg, feats):
        feat_list = []
        atom_list = []
        # cal specific feats
        for i in range(self.task_num):
            with bg.local_scope():
                bg.ndata['h'] = feats
                weight = self.atom_weighting_specific[i](feats)
                bg.ndata['w'] = weight
                specific_feats_sum = sum_nodes(bg, 'h', 'w')
                atom_list.append(bg.ndata['w'])
            feat_list.append(specific_feats_sum)

        # cal shared feats
        with bg.local_scope():
            bg.ndata['h'] = feats
            bg.ndata['w'] = self.shared_weighting(feats)
            shared_feats_sum = sum_nodes(bg, 'h', 'w')
        # feat_list.append(shared_feats_sum)
        if self.attention:
            if self.return_weight:
                return feat_list, atom_list
            else:
                return feat_list
        else:
            return shared_feats_sum

    def atom_weight(self, in_feats):
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
            )

class MTGL_ADMET(nn.Module):
    def __init__(self, in_feats,hidden_feats,gnn_out_feats=64,n_tasks=None,  return_weight=False,
                 classifier_hidden_feats=128, dropout=0.):
        super(MTGL_ADMET, self).__init__()


        self.task_num = n_tasks
        self.return_weight = return_weight
        self.weighted_sum_readout = WeightAndSum(gnn_out_feats, self.task_num, return_weight=self.return_weight)

        # Two-layer GCN
        self.conv1 = ResGCNLayer(in_feats, hidden_feats)
        self.conv2 = ResGCNLayer(hidden_feats, gnn_out_feats)

        self.gates = nn.ModuleList()
        for i in range(self.task_num):
            self.gates.append(nn.Linear(64, 2))

        self.fc_in_feats = gnn_out_feats
        for i in range(self.task_num):
            self.fine_f = nn.ModuleList([self.fc_layer(dropout,gnn_out_feats, gnn_out_feats) for _ in range(self.task_num)])

        self.fc_layers1 = nn.ModuleList([self.fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.output_layer1 = nn.ModuleList(
            [self.output_layer(classifier_hidden_feats, 1) for _ in range(self.task_num)])

    def forward(self, bg, node_feats):

        node_feats = self.conv1(bg, node_feats)
        node_feats = self.conv2(bg, node_feats)

        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)

        # Number of Auxiliary tasks
        num_gates = 4
        # gate input
        combine = []
        bg.ndata['h'] = node_feats
        hg = dgl.mean_nodes(bg, 'h')

        for i in range(num_gates):

            auxi = feats_list[i]
            auxi_u = torch.unsqueeze(auxi, dim=1)
            prim = feats_list[num_gates]
            prim_u = torch.unsqueeze(prim,dim=1)
            gating_f = torch.cat((auxi_u, prim_u), dim=1)
            gate = self.gates[i](hg)
            gate = F.softmax(gate, dim=-1)
            gate = torch.unsqueeze(gate, dim=-1)
            gating_r = torch.sum(gating_f * gate, dim=1)
            combine.append(gating_r)
        gating_combine = combine[0]+combine[1]+combine[2]+combine[3]

        combine_2 = []
        combine_2.append(feats_list[0])
        combine_2.append(gating_combine)
        combine_2.append(feats_list[1])
        combine_2.append(feats_list[2])
        combine_2.append(feats_list[3])

        # FC
        for i in range(self.task_num):
            mol_feats = combine_2[i]
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            predict = self.output_layer1[i](h2)
            if i == 0:
                prediction_all = predict
            else:
                prediction_all = torch.cat([prediction_all, predict], dim=1)

        return prediction_all


    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats)
            )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
            )
    def atom_weight(self, in_feats):
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
            )





