import torch
from torch import nn
import math
from model.modules import ConvSC
from model.modules.layers import (Attention, MixMlp)
from timm.models.layers import DropPath
import copy


class TCU(nn.Module):

    def __init__(self, seqlen_in, seqlen_out, channel_in, channel_out, H, W, hid_S=4, hid_T=4, N_S=2, N_T=2, model_type='gSTA',
                 mlp_ratio=2., drop=0.0, drop_path=0.0, spatio_kernel_enc=1,
                 spatio_kernel_dec=2, act_inplace=True, **kwargs):
        super(TCU, self).__init__()

        act_inplace = False
        self.seqlen_in = seqlen_in
        self.seqlen_out = seqlen_out
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.H = H
        self.W = W
        self.hid_S = hid_S

        self.enc = Encoder(channel_in, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, channel_out, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.hid = TCUSubBlock(seqlen_in*hid_S, seqlen_out*hid_S, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x_raw, **kwargs):
        B = x_raw.shape[0]
        x = x_raw.view(B*self.seqlen_in, self.channel_in, self.H, self.W)
        embed = self.enc(x)
        z = embed.view(B, self.seqlen_in, self.hid_S, self.H, self.W)
        hid = self.hid(z)
        hid = hid.reshape(B*self.seqlen_out, self.hid_S, self.H, self.W)
        Y = self.dec(hid)
        Y = Y.reshape(B, self.seqlen_out, self.channel_out, self.H, self.W)
        return Y


class Encoder(nn.Module):
    """3D Encoder for SimVP"""
    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        super(Encoder, self).__init__()
        self.enc = ConvSC(C_in, C_hid, spatio_kernel, downsampling=False, act_inplace=act_inplace)

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc(x)
        return enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        super(Decoder, self).__init__()
        self.dec = ConvSC(C_hid, C_hid, spatio_kernel, upsampling=False, act_inplace=act_inplace)
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid):
        Y = self.dec(hid)
        Y = self.readout(Y)
        return Y


class TCUBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=1., drop=0., drop_path=0., init_value=1e-2, act_layer=nn.GELU, attn_shortcut=True):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, attn_shortcut=attn_shortcut)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def _init_weights(self, m):
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class TCUSubBlock(nn.Module):
    def __init__(self, dim, out_dim, mlp_ratio=1., drop=0., drop_path=0., init_value=1e-2, act_layer=nn.GELU):
        super().__init__()
        self.extractor = TCUBlock(dim=dim, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, init_value=init_value, act_layer=act_layer)
        self.target_extractor = copy.deepcopy(self.extractor).requires_grad_(False)
        self.zeros = nn.Parameter(torch.zeros(1, 1, 1, 1, 1))
        self.zeros.requires_grad_ = False
        self.readout = nn.Conv2d(dim, out_dim, 1)

    def soft_update(self):
        for target, source in zip(self.target_extractor.parameters(), self.extractor.parameters()):
            target.data.mul_(0.995).add_(source.data, alpha=0.005)

    def forward(self, x):
        b, t, c, h, w = x.size()
        input = x.view(b, t*c, h, w)
        out = self.extractor(input)
        out = out.view(b, t, c, h, w)

        delta = x[:, 1:t] - x[:, :t-1]  # b, t-1, c, h, w
        delta_input_ = torch.cat([delta, self.zeros.repeat(b, 1, c, h, w)], dim=1)
        delta_out_ = self.target_extractor(delta_input_.view(b, t*c, h, w))
        delta_input_reverse = torch.cat([self.zeros.repeat(b, 1, c, h, w), delta], dim=1)
        delta_out_reverse = self.target_extractor(delta_input_reverse.view(b, t*c, h, w))
        delta_out_ = delta_out_.view(b, t, c, h, w)
        delta_out_reverse = delta_out_reverse.view(b, t, c, h, w)

        x = out.clone()
        x[:, 0] = 0.95*x[:, 0] + 0.05*(out[:, 1] - delta_out_reverse[:, 0])
        x[:, 1:t-1] = 0.9*x[:, 1:t-1] + 0.05*(out[:, 2:t] - delta_out_reverse[:, 2:t]) + 0.05*(out[:, 0:t-2] + delta_out_[:, 0:t-2])
        x[:, t-1] = 0.95*x[:, t-1] + 0.05*(out[:, t-2] + delta_out_[:, t-2])
        self.soft_update()
        x = x.view(b, t*c,  h,  w)
        x = self.readout(x)
        return x.view(b, -1, c, h, w)
