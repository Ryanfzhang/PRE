import numpy as np
import torch
from torch import nn
from linear_attention_transformer import LinearAttentionTransformer
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from torch.nn import Parameter
from einops import rearrange
from math import sqrt

device = "cuda" if torch.cuda.is_available() else "cpu"

class TokenEmbedding(nn.Module): 
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionEmbedding_v2(nn.Module):
    def __init__(self, d_model, is_sea, mean, std):
        super(PositionEmbedding_v2, self).__init__()
        self.d_model = d_model
        self.mean = mean
        self.std = std
        self.is_sea = is_sea
        learnable_position_embedding = self.get_position_embeding()[:,is_sea.bool()]
        self.projection1 = nn.Linear(3, d_model)
        self.register_buffer("embedding", learnable_position_embedding)

    def forward(self):
        x = self.embedding.transpose(0,1)
        x = self.projection1(x)
        x = x.transpose(0,1)
        x = x.unsqueeze(0).unsqueeze(0)
        return x

    def get_position_embeding(self):
        height = 60
        width = 96
        mean = torch.zeros(height, width)
        std = torch.zeros(height, width)
        mean[self.is_sea.bool()] = self.mean.cpu()
        std[self.is_sea.bool()] = self.std.cpu()
        pos_w = torch.arange(0., width)/width
        pos_h = torch.arange(0., height)/height
        pos_w = pos_w.unsqueeze(0).expand(height, -1)
        pos_h = pos_h.unsqueeze(1).expand(-1, width)
        pe = torch.stack([mean, std, pos_h], 0)
        pe = pe.to(device)
        return pe

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B,H,-1,-1)
            scores.masked_fill_(attn_mask.bool(), -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

# class Inception_Block_V1(nn.Module):
#     def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
#         super(Inception_Block_V1, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_kernels = num_kernels
#         kernels = []
#         for i in range(self.num_kernels):
#             kernels.append(nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
#         self.kernels = nn.ModuleList(kernels)
#         if init_weight:
#             self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         res_list = []
#         for i in range(self.num_kernels):
#             res_list.append(self.kernels[i](x))
#         res = torch.stack(res_list, dim=-1).mean(-1)
#         return res


class GCN(nn.Module):
    def __init__(self,
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 c_hid,
                 num_types,
                 temp=1, # temperature parameter
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.num_types = num_types
        self.temp = temp

        # Initialization
        nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        self.weights_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(c_hid, c_in, c_out)))

    def forward(self,
                node_feats, # input node features
                adj_matrix, # adjacency matrix including self-connections
                position_embedding
                ):

        # Apply linear layer and sort nodes by head
        node_feats = torch.matmul(adj_matrix, node_feats)
        position_embedding = torch.matmul(adj_matrix, position_embedding)
        position_weights = torch.einsum('nd, dio-> nio', position_embedding, self.weights_pool)
        node_feats = torch.einsum('bni, nio->bno', node_feats, position_weights)
        return node_feats

class GraphTransformer(nn.Module):
    def __init__(self, config, is_sea, mean, std):
        super(type(self), self).__init__()
        self.config = config 
        self.c_in = 1
        self.c_out= 1
        self.c_hid = config['hidden_dim']
        self.out_len = config['out_len']
        self.in_len = config['in_len']
        self.norm1 = nn.LayerNorm(self.c_hid)
        self.norm2 = nn.LayerNorm(self.c_hid)
        self.gn = nn.GroupNorm(4, self.c_hid)

        self.value_embedding = TokenEmbedding(c_in=self.c_in, d_model=self.c_hid)
        self.position_embedding = PositionEmbedding_v2(d_model=self.c_hid, is_sea=is_sea, mean=mean, std=std)

        self.spatial_encoder = GCN(self.c_hid, self.c_hid, self.c_hid, 3)
        # self.temporal_encoder = AttentionLayer(FullAttention(False, attention_dropout=0, output_attention=False), self.c_hid, 1)
        self.temporal_encoder = LinearAttentionTransformer(dim=self.c_hid, depth=1, heads=1, max_seq_len=100, n_local_attn_heads=0, local_attn_window_size=0)

        # self.projection = nn.Conv1d(self.c_hid, self.c_out, kernel_size= 3, padding=1)
        self.projection1 = nn.Linear(self.in_len, self.out_len)
        self.projection2 = nn.Linear(self.c_hid, self.c_out)

    def forward(self, x_enc, adj):
        B, L, D, N = x_enc.shape
        mean = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean
        x_enc = rearrange(x_enc, 'b l d n -> (b n) l d')
        enc_out = self.value_embedding(x_enc)
        position_embedding = self.position_embedding()
        enc_out = rearrange(enc_out, '(b n) l d -> b l d n', b=B, n=N)
        # enc_out = enc_out + position_embedding
        position_embedding = position_embedding.squeeze()

        D = enc_out.shape[-1]
        enc_out = rearrange(enc_out, 'b l d n -> (b n) l d', b=B, n=N)
        enc_out = self.temporal_encoder(enc_out)
        enc_out = rearrange(enc_out, '(b n) l d -> (b l) n d', b=B, n=N)

        enc_out = self.spatial_encoder(enc_out, adj, position_embedding.transpose(0,1))
        enc_out = self.gn(enc_out.transpose(1,2))
        enc_out = rearrange(enc_out, '(b l) d n -> (b n) l d', b=B, l=L)

        out = self.projection2(enc_out)
        out = F.silu(out)
        out = self.projection1(out.permute(0,2,1))
        out = rearrange(out, '(b n) d l -> b l d n', b=B, n=N)
        out = out + mean
        return out

