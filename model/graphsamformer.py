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
        self.time_encoding = LinearAttentionTransformer(dim=self.hidden_dim, depth=1, heads=1, max_seq_len=16, n_local_attn_heads=0, local_attn_window_size=0)
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
        x = self.normalize(x)
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

        x = self.mid_projection(x)
        gate, filter = torch.chunk(x, 2, dim=1)
        y = torch.sigmoid(gate)*torch.tanh(filter)
        y = self.out_projection(y)
        y = rearrange(y, '(b t) c n -> (b n) t c', b=B, t=T)
        y = F.silu(y)
        y = self.predictor(y)
        y = rearrange(y, '(b n) t c -> b t c n', b=B, n=N)
        y = self.denormalize(y)
        return y

    def normalize(self, x):
        mean = self.mean.reshape(1,1,1,-1)
        std = self.std.reshape(1,1,1,-1)
        normalized_x = (x - mean)/(std+1e-8)
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

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """
    A copy-paste from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / np.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class SAM(Optimizer):
    """
    SAM: Sharpness-Aware Minimization for Efficiently Improving Generalization https://arxiv.org/abs/2010.01412
    https://github.com/davda54/sam
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm