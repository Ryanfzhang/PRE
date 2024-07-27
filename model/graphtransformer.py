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


class moving_avg(nn.Module):
    """ Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=300):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

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

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

    def forward(self, x):
        x = self.value_embedding(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention( x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

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

    def forward(self, queries, keys, values, attn_mask):
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

        # if self.mask_flag:
        #     if attn_mask is None:
        #         attn_mask = TriangularCausalMask(B, L, device=queries.device)

        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # x = x + self.dropout(self.self_attention(
        #     x, x, x,
        #     attn_mask=x_mask
        # )[0])
        # x, trend1 = self.decomp1(x)
        # x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        y = x = self.norm2(x)

        y = x 
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))


        return self.norm3(x + y)

class GraphTransformer(nn.Module):
    def __init__(self, config):
        super(type(self), self).__init__()
        self.config = config
        self.c_in = 1
        self.c_out= 1
        self.c_hid = config['hidden_dim']
        self.out_len = config['out_len']
        self.norm = nn.LayerNorm(self.c_hid)
        self.enc_embedding = DataEmbedding(1, self.c_hid)
        self.dec_embedding = DataEmbedding(1, self.c_hid)
        self.weights_pool = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(4, self.c_hid, self.c_hid)))
        self.position_embedding = nn.Parameter(torch.randn(4443, 4), requires_grad=True)
        self.encoder = EncoderLayer(AttentionLayer(FullAttention(False, attention_dropout=0.1, output_attention=True), self.c_hid, 1), self.c_hid, self.c_hid, dropout=0)
        self.decoder = DecoderLayer(AttentionLayer(FullAttention(True, attention_dropout=0.1, output_attention=False), self.c_hid, 1), AttentionLayer(FullAttention(False, attention_dropout=0.1, output_attention=False), self.c_hid, 1), self.c_hid, self.c_out, dropout=0.1)
        self.projection =nn.Linear(self.c_hid, self.c_out, bias=True)

    def forward(self, x_enc, x_dec, adj_matrix):
        B, L, D, N = x_enc.shape
        _, L2, _, _ = x_dec.shape
        # adj1 = torch.matmul(adj_matrix, adj_matrix)
        # adj = torch.matmul(adj1, adj_matrix)
        # adj_mask = torch.eye(adj.shape[0], device=adj.device)
        # norm = (adj*adj_mask).sum(-1, keepdim=True)
        # adj = adj/(norm+1e-4)

        mean = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean
        x_enc = rearrange(x_enc, 'b l d n -> (b n) l d')
        enc_out = self.enc_embedding(x_enc)
        D = enc_out.shape[-1]
        enc_out = rearrange(enc_out, '(b n) l d -> n (b l d)', b=B, n=N)
        # enc_out = torch.matmul(adj_matrix, enc_out)
        enc_out = rearrange(enc_out,'n (b l d) -> (b n) l d', b=B, l=L, d=D)
        # enc_out = self.norm(enc_out)
        enc_out, attns = self.encoder(enc_out)

        x_dec = rearrange(x_dec, 'b l d n -> (b n) l d')
        dec_mean = x_dec[:,:self.out_len].mean(1, keepdim=True)
        x_dec[:,:self.out_len] = x_dec[:,:self.out_len] - dec_mean
        dec_out = self.dec_embedding(x_dec)
        dec_out = rearrange(dec_out, '(b n) l d -> n (b l d)', b=B, n=N)
        # dec_out = torch.matmul(adj_matrix, dec_out)
        dec_out = rearrange(dec_out,'n (b l d) -> (b n) l d', b=B, l=3*L//2, d=D)
        # dec_out = self.norm(dec_out)

        dec_out = self.decoder(dec_out, enc_out)
        dec_out = self.projection(dec_out)

        dec_out = rearrange(dec_out, '(b n) l d -> b l d n', b=B, n=N)
        dec_out = dec_out + mean
        return dec_out[:, -self.out_len:, :, :]

