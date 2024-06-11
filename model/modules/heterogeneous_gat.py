import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DenseLayer(nn.Module):
    def __init__(self,
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 zero_init=False, # initialize weights as zeros; use Xavier uniform init if zero_init=False
                 ):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out)
        # Initialization
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self,
    			node_feats, # input node features
    			):
        node_feats = self.linear(node_feats)
        return node_feats

class GATSingleHead(nn.Module):
    def __init__(self,
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 temp=1, # temperature parameter
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out)
        self.v0 = nn.Parameter(torch.Tensor(c_out, 1))
        self.v1 = nn.Parameter(torch.Tensor(c_out, 1))
        self.temp = temp

        # Initialization
        nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)
        nn.init.uniform_(self.v0.data, -np.sqrt(6 / (c_out + 1)), np.sqrt(6 / (c_out + 1)))
        nn.init.uniform_(self.v1.data, -np.sqrt(6 / (c_out + 1)), np.sqrt(6 / (c_out + 1)))

    def forward(self,
                node_feats, # input node features
                adj_matrix, # adjacency matrix including self-connections
                ):

        # Apply linear layer and sort nodes by head
        node_feats = self.linear(node_feats)
        f1 = torch.matmul(node_feats, self.v0)
        f2 = torch.matmul(node_feats, self.v1)
        attn_logits = adj_matrix * (f1 + f2.T)
        unnormalized_attentions = (F.sigmoid(attn_logits) - 0.5).to_sparse()
        attn_probs = torch.sparse.softmax(unnormalized_attentions / self.temp, dim=1)
        attn_probs = attn_probs.to_dense()
        node_feats = torch.matmul(attn_probs, node_feats)

        return node_feats

class HGAT(nn.Module):
    def __init__(self, c_in, c_out, num_types):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.num_types = num_types
        self.projection = DenseLayer(c_in, c_out)
        self.k_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()

        for _ in range(self.num_types):
            self.k_linears.append(DenseLayer(c_out, 1))
            self.v_linears.append(DenseLayer(c_out, 1))

    def forward(self, node_rep, adj_matrix, node_type):
        # node_rep b, n, c
        # adj n, n
        # node_type n, 1
        B = node_rep.shape[0]
        adj_matrix = adj_matrix.unsqueeze(0).expand(B,-1,-1)
        node_type = node_type.unsqueeze(0).unsqueeze(-1).expand_as(node_rep)

        node_rep = self.projection(node_rep)
        ks = torch.cat([self.k_linears[i](node_rep) for i in range(self.num_types)], dim=-1)
        vs = torch.cat([self.v_linears[i](node_rep) for i in range(self.num_types)], dim=-1)
        f1 = torch.gather(ks, 2, node_type)
        f2 = torch.gather(vs, 2, node_type)

        attn_logits = adj_matrix * (f1 + f2.T)
        unnormalized_attentions = (F.sigmoid(attn_logits) - 0.5)
        attn_probs = torch.softmax(unnormalized_attentions / self.temp, dim=1)
        node_rep = torch.matmul(attn_probs, node_rep)

        return node_rep
