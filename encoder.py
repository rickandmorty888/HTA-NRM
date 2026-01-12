import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_bands=6, d_model=768):
        super().__init__()
        freq = 2.0 ** torch.linspace(0, num_bands - 1, num_bands)
        self.freq_bands = nn.Parameter(freq)
        self.proj = nn.Linear(3 * num_bands * 2 * d_model, d_model, bias=False)

    def forward(self, xyz):
        B, _, N, D = xyz.shape
        xb = (2 * np.pi) * xyz.unsqueeze(-1) * self.freq_bands
        feat = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)
        feat = feat.view(B, N, -1)
        return self.proj(feat)


class VoxelMixer(nn.Module):
    def __init__(self, token_dim=768, tokens=21, depth=4):
        super().__init__()

        def mlp(dim):
            return nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            )

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(nn.LayerNorm(token_dim), mlp(token_dim)))
            self.blocks.append(nn.Sequential(nn.LayerNorm(tokens), mlp(tokens)))

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:
                x = x + blk(x)
            else:
                x = x + blk(x.transpose(1, 2)).transpose(1, 2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model=768, n_subj=4):
        super().__init__()
        self.ntok = 21
        self.d_model = d_model

        self.reg_pos = nn.Parameter(torch.zeros(n_subj, self.ntok, d_model))
        self.pe = FourierPositionalEncoding()

        self.semantic = VoxelMixer()

        self.q_low = nn.Linear(d_model, 8 * d_model)
        self.q_mid = nn.Linear(d_model, 4 * d_model)
        self.q_high = nn.Linear(d_model, 2 * d_model)

        self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)

    def forward(self, x, subj_idx, xyz):
        x = self.semantic(x)
        x = x + self.reg_pos[subj_idx] + self.pe(xyz)

        q = x.mean(dim=1)
        z_low = self.q_low(q).view(x.size(0), 8, self.d_model)
        z_mid = self.q_mid(q).view(x.size(0), 4, self.d_model)
        z_high = self.q_high(q).view(x.size(0), 2, self.d_model)

        return z_low, z_mid, z_high
