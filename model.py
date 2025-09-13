import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class SingleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.act(self.conv(x))


def center_crop(tensor, target_hw):
    _, _, h, w = tensor.shape
    th, tw = target_hw
    top = (h - th) // 2
    left = (w - tw) // 2
    return tensor[..., top:top+th, left:left+tw]


class UnetSrplus(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, depth=5, base_channels=32, upscale_factor=1):
        super().__init__()

        self.depth = depth
        self.base_channels = base_channels
        self.upscale_factor = int(upscale_factor)
        self.pool = nn.MaxPool2d(2, 2)

        # encoder
        enc = []
        ch = base_channels
        in_c = in_ch
        for i in range(depth):
            enc.append(SingleBlock(in_c, ch))
            in_c = ch
            ch = ch * 2
        self.enc_blocks = nn.ModuleList(enc)

        # bottleneck
        self.bottleneck = SingleBlock(in_c, ch)

        # decoder
        up_convs = []
        dec = []
        ch_decoder = ch
        for i in range(depth):
            up_convs.append(nn.ConvTranspose2d(ch_decoder, ch_decoder // 2, kernel_size=2, stride=2))
            dec.append(SingleBlock(ch_decoder, ch_decoder // 2))
            ch_decoder = ch_decoder // 2
        self.up_convs = nn.ModuleList(up_convs)
        self.dec_blocks = nn.ModuleList(dec)

        # final output conv
        self.final_conv = nn.Conv2d(base_channels, out_ch, kernel_size=1)
        nn.init.kaiming_normal_(self.final_conv.weight, nonlinearity='linear')
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x):
        if self.upscale_factor > 1:
            x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        skips = []
        cur = x
        # encoder
        for enc in self.enc_blocks:
            cur = enc(cur)
            skips.append(cur)
            cur = self.pool(cur)

        # bottleneck
        cur = self.bottleneck(cur)

        # decoder
        for up, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            cur = up(cur)
            if cur.shape[-2:] != skip.shape[-2:]:
                skip = center_crop(skip, cur.shape[-2:])
            cur = torch.cat([cur, skip], dim=1)
            cur = dec(cur)

        out = self.final_conv(cur)
        return out
