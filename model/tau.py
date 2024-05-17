import torch
from torch import nn
import math
import copy
from timm.models.layers import DropPath, trunc_normal_
from model.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock, TCUSubBlock)
from model.modules.layers import (HorBlock, ChannelAggregationFFN, MultiOrderGatedAggregation, Attention,
                     PoolFormerBlock, CBlock, SABlock, MixMlp, VANBlock)


class TAU(nn.Module):

    def __init__(self, seqlen_in, seqlen_out, channel_in, channel_out, H, W, hid_S=4, hid_T=4, N_S=1, N_T=1, model_type='gSTA',
                 mlp_ratio=2., drop=0.0, drop_path=0.0, spatio_kernel_enc=1,
                 spatio_kernel_dec=2, act_inplace=True, **kwargs):
        super(TAU, self).__init__()

        act_inplace = False
        self.seqlen_in= seqlen_in 
        self.seqlen_out= seqlen_out
        self.channel_in = channel_in
        self.channel_out= channel_out
        self.H = H
        self.W = W
        self.hid_S = hid_S

        self.enc = Encoder(channel_in, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, channel_out, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.hid = TemporalAttentionModule(seqlen_in*hid_S, seqlen_out*hid_S, kernel_size=3)

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
        self.enc = ConvSC(C_in, C_hid, spatio_kernel, downsampling=False, 
                     act_inplace=act_inplace)

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc(x)
        return enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        super(Decoder, self).__init__()
        self.dec = ConvSC(C_hid, C_hid, spatio_kernel, upsampling=False,
                     act_inplace=act_inplace) 
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid):
        Y = self.dec(hid)
        Y = self.readout(Y)
        return Y

class TemporalAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, out_dim, kernel_size, dilation=1, reduction=1):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
            nn.Sigmoid()
        )
        self.readout = nn.Conv2d(dim, out_dim, 1)

    def forward(self, x):
        b,t,c,h,w = x.shape
        x = x.view(b,t*c,h,w)
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        f_x = self.conv1(attn)         # 1x1 conv
        # append a se operation
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return self.readout(se_atten * f_x * u)
