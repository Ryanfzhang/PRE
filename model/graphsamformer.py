import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer
from einops import rearrange, repeat
from torch.optim import Optimizer
import numpy as np

class GraphFormer(nn.Module):
    def __init__(self, config, mean, std):
        super().__init__()
        self.seq_in = config['in_len']
        self.seq_out = config['out_len']
        self.channel_in = 1
        self.hidden_dim = config['hidden_dim']

        self.input_projection = Conv1d_with_init(1, self.hidden_dim, 1)
        self.time_encoding = LinearAttentionTransformer(dim=self.hidden_dim, depth=1, heads=1, max_seq_len=46, n_local_attn_heads=0, local_attn_window_size=0)
        self.spatial_encoding = GCN(self.hidden_dim, self.hidden_dim, 3)
        self.mid_projection = Conv1d_with_init(self.hidden_dim, 2*self.hidden_dim, 1)
        self.out_projection = Conv1d_with_init(self.hidden_dim, self.channel_in, 1)
        self.predictor= Conv1d_with_init(self.seq_in, self.seq_out, 1)
        self.gn = nn.GroupNorm(4, self.hidden_dim)

        self.mean = mean
        self.std = std
    def forward(self, x, adj, node_type):
        # x [b, t, c, n]
        B, T, _, N = x.shape
        x = rearrange(x, 'b t c n->(b t) c n')
        x = self.input_projection(x)
        C = x.shape[1]

        x = rearrange(x, '(b t) c n -> (b n) t c', b=B, t=T)
        x = self.time_encoding(x)
        x = rearrange(x, '(b n) t c -> (b t) n c', b=B, n=N)

        x_in = x
        x = self.spatial_encoding(x, adj, node_type)
        x = rearrange(x, '(b t) n c-> (b t) c n', b=B, t=T)
        x_in = rearrange(x_in, '(b t) n c-> (b t) c n', b=B, t=T)
        x = x + x_in
        x = self.gn(x)

        # x = self.mid_projection(x)
        # gate, filter = torch.chunk(x, 2, dim=1)
        # y = torch.sigmoid(gate)*torch.tanh(filter)
        y = self.out_projection(x)
        y = rearrange(y, '(b t) c n -> (b n) t c', b=B, t=T)
        y = F.gelu(y)
        y = self.predictor(y)
        y = rearrange(y, '(b n) t c -> b t c n', b=B, n=N)
        return y

    def normalize(self, x):
        mean = self.mean.reshape(1,1,1,-1)
        std = self.std.reshape(1,1,1,-1)
        normalized_x = (x - mean)/(std + 1e-6)
        return normalized_x

    def denormalize(self, x):
        mean = self.mean.reshape(1,1,1,-1)
        std = self.std.reshape(1,1,1,-1)
        denormalized_x = x*std + mean
        return denormalized_x

class GCN(nn.Module):
    def __init__(self,
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 num_types,
                 temp=1, # temperature parameter
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.num_types = num_types
        self.temp = temp

        # Initialization
        nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))

    def forward(self,
                node_feats, # input node features
                adj_matrix, # adjacency matrix including self-connections
                node_type,
                ):

        # Apply linear layer and sort nodes by head
        node_feats = torch.matmul(adj_matrix, node_feats)
        node_feats = self.linear(node_feats)
        return node_feats

# class GCN(nn.Module):
#     def __init__(self,
#                  c_in, # dimensionality of input features
#                  c_out, # dimensionality of output features
#                  num_types,
#                  temp=1, # temperature parameter
#                  ):

#         super().__init__()

#         # self.linear = nn.Linear(c_in, c_out, bias=False)
#         # self.linear_bias = nn.Linear(16, c_out, bias=False)
#         self.num_types = num_types
#         self.temp = temp
#         # self.bias_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(8, c_out)))
#         self.weights_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(16, c_in, c_out)))
#         self.position_embedding = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(4443, 16)))


#         # Initialization
#         # nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
#         # nn.init.uniform_(self.linear_bias.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))

#     def forward(self,
#                 node_feats, # input node features
#                 adj_matrix, # adjacency matrix including self-connections
#                 node_type,
#                 ):

#         # Apply linear layer and sort nodes by head
#         node_feats = torch.matmul(adj_matrix, node_feats)
#         position_embedding = torch.matmul(adj_matrix, self.position_embedding)
#         position_embedding = torch.matmul(adj_matrix, position_embedding)
#         position_embedding = torch.matmul(adj_matrix, position_embedding)
#         position_weights = torch.einsum('nd, dio-> nio', position_embedding, self.weights_pool)
#         node_feats = torch.einsum('bni, nio->bno', node_feats, position_weights)
#         return node_feats

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer