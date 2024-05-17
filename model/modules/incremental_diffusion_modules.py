import numpy as np
import torch
from torch import nn
from linear_attention_transformer import LinearAttentionTransformer
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None): 
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class STmodule(nn.Module):
    def __init__(self, config, input_dim):
        self.config = config
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.input_dim = input_dim
        self.patch_size = self.config['patch_size']

        self.diffusion_embedding = DiffusionEmbedding(num_steps=config['num_steps'], embedding_dim=config['hidden_channels'])
        self.position_embedding = self.get_position_embeding()

        self.input_projection = nn.Conv1d(input_dim, config['hidden_channels'], 1)
        nn.init.kaiming_normal_(self.input_projection.weight)

        self.time_encoding = LinearAttentionTransformer(dim=self.config['hidden_channels'], depth=1, heads=1, max_seq_len=256, n_local_attn_heads=0, local_attn_window_size=0)
        self.patchify = torch.nn.Conv2d(self.config['hidden_channels'], self.config['hidden_channels'], self.patch_size, self.patch_size)
        self.spatial_encoding = Block(self.config['hidden_channels'], self.config['num_heads'])
        self.head = torch.nn.Linear(self.config['hidden_channels'], self.patch_size**2)

    def forward(self, x, cond_mask, diffusion_step, cond_ob, low_resolution=None): 
        if low_resolution==None:
            input = torch.stack([x, cond_ob], dim=3)
        else:
            input = torch.stack([x, cond_ob, low_resolution], dim=3)

        B, T, K, C, H, W = x.shape
        assert C==self.input_dim, print("input dim is not match the pre definition")

        input = rearrange(input, "b t k c h w -> (b t k) c (h w)")
        x = self.input_projection(input)
        x = rearrange(x, "(b t k) c (h w) -> b t k c h w", b=B, t=T, k=K, h=H, w=W)

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        diffusion_emb = diffusion_emb.unsqueeze(1).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        x = x + diffusion_emb

        x = rearrange(x, 'b t k c h w -> (b k h w) t c')
        x = self.time_encoding(x)
        x = rearrange(x, '(b k h w) t c -> (b t k) c h w', b=B, t=T, k=K, c=C, h=H, w=W)
        patches = self.patchify(input) # (B T K) C H//patch_size W//patch_size
        patches = patches + self.position_embedding.unsqueeze(0)
        patches = rearrange(patches, '(b t k) c h w -> (b t k) (h w) c')
        features = self.spatial_encoding(patches)

        predicted = self.head_sp(features) # (B T K) H//patch_size, W//patch_size, patch_size**2
        predicted = rearrange(predicted, "(b t k) (h w) (p1 p2)-> b t k (h p1) (w p2)", b=B, t=T, k=K, h=self.config["height"]//self.patch_size, w=self.config['width']//self.patch_size, p1=self.patch_size, p2=self.patch_size)

        return predicted

    def get_time_embedding(self):
        pe = torch.zeros(self.config['in_len'], self.config['hidden_channels'], self.config['height'], self.config['width']).to(self.device)
        position = torch.arange(self.config['in_len']).unsqueeze(1).to(self.device)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, self.config['hidden_channels'], 2).to(self.device) / self.config['hidden_channels']
        )
        pe[:, 0::2, :, :] = torch.sin(position * div_term).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.config['height'], self.config['width'])
        pe[:, 1::2, :, :] = torch.cos(position * div_term).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.config['height'], self.config['width'])
        return pe # T, Hidden, H, W

    def get_position_embeding(self):
        pe = torch.zeros(self.config['hidden_channels'], self.config['height'], self.config['width'])
        d_model = int(self.config['hidden_channels'] / 2)
        height = self.config['height']//self.patch_size
        width = self.config['width']//self.patch_size
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe = pe.to(self.device)

        return pe # Hidden, H, W
