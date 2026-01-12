import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, d_model=768, seq_out=21, hidden=768, rank=16,
                 num_heads=8, layers: int = 2, drop: float = 0.1):
        super().__init__()
        self.seq_out, self.rank = seq_out, rank
        self.layers = layers

        # ---- fixed 21-grid (7x3) ----
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, 7), torch.linspace(-1, 1, 3), indexing="ij"
        ), -1).reshape(seq_out, 2)
        self.register_buffer("grid_xy", grid)
        self.coord_proj = nn.Linear(2, d_model, bias=False)

        # learnable grid tokens
        self.grid_kv = nn.Parameter(torch.randn(1, seq_out, d_model))
        nn.init.trunc_normal_(self.grid_kv, std=0.02)

        # ---- latent-conditioned decoding (NEW) ----
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.xattn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=drop)
        # small residual scale for stability (like your LayerScale idea)
        self.ls_dec = nn.Parameter(torch.tensor(1e-3))

        # (optional) light FFN after attention, still very small effect at init
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_model),
        )
        self.ls_ffn = nn.Parameter(torch.tensor(1e-3))

        # ---- heads (unchanged) ----
        self.head_raw   = nn.Linear(d_model, d_model)
        self.head_voxel = nn.Linear(d_model, d_model * 3)
        self.head_region_wide = nn.Linear(d_model, 256, bias=True)
        self.region_reduce = nn.Linear(256, rank, bias=False)

        self.scale_voxel  = nn.Parameter(torch.zeros(1))  # scalar
        self.scale_region = nn.Parameter(torch.zeros(1))  # scalar, keep symmetry
        self.register_buffer("eyeS", torch.eye(self.seq_out))

    def forward(self, z_low):
        """
        z_low: (B, K_low=8, 768)   # your encoder output z_out['z_low']
        return:
            voxel : (B, 21, 768, 3)
            region: (B, 21, 21)
            raw   : (B, 21, 768)
        """
        B = z_low.size(0)

        # base grid tokens
        kv0 = self.grid_kv.expand(B, -1, -1) + self.coord_proj(self.grid_xy).unsqueeze(0)
        h = kv0  # (B,21,768)

        # ---- NEW: decode with latent slots ----
        # Query = grid tokens, Key/Value = z_low
        for _ in range(self.layers):
            attn_out, _ = self.xattn(self.ln_q(h), self.ln_kv(z_low), self.ln_kv(z_low))
            h = h + self.ls_dec * attn_out

            h = h + self.ls_ffn * self.ffn(h)

        # heads
        raw = self.head_raw(h)  # (B,21,768)

        voxel = self.head_voxel(h).view(B, self.seq_out, 768, 3) * F.softplus(self.scale_voxel)

        Lw = self.head_region_wide(h)          # (B,21,256)
        L = self.region_reduce(Lw)             # (B,21,rank)
        region = torch.bmm(L, L.transpose(1, 2)) / self.rank  # (B,21,21) PSD
        region = region * F.softplus(self.scale_region)
        region = region + 1e-4 * self.eyeS.to(region.dtype).unsqueeze(0)

        return voxel, region, raw
