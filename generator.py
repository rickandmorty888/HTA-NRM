import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Topology-aware generative decoder.

    Given low-level latent representations from the encoder, this module
    reconstructs:
        1) voxel-wise spatial topology
        2) inter-region connectivity (PSD matrix)
        3) raw neural feature representations

    The decoder operates on a fixed 21-region grid and uses cross-attention
    to inject latent semantic information.
    """

    def __init__(
        self,
        d_model: int = 768,
        seq_out: int = 21,
        hidden: int = 768,
        rank: int = 16,
        num_heads: int = 8,
        layers: int = 2,
        drop: float = 0.1,
    ):
        super().__init__()

        self.seq_out = seq_out
        self.rank = rank
        self.layers = layers

        # ------------------------------------------------------------------
        # Fixed 2D grid (7 × 3) for 21 cortical regions
        # ------------------------------------------------------------------
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, 7),
                torch.linspace(-1, 1, 3),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(seq_out, 2)

        self.register_buffer("grid_xy", grid)
        self.coord_proj = nn.Linear(2, d_model, bias=False)

        # Learnable grid tokens
        self.grid_kv = nn.Parameter(torch.randn(1, seq_out, d_model))
        nn.init.trunc_normal_(self.grid_kv, std=0.02)

        # ------------------------------------------------------------------
        # Latent-conditioned cross-attention decoder
        # ------------------------------------------------------------------
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)

        self.xattn = nn.MultiheadAttention(
            d_model,
            num_heads,
            batch_first=True,
            dropout=drop,
        )

        # LayerScale-style small residual for stability
        self.ls_dec = nn.Parameter(torch.tensor(1e-3))

        # Lightweight FFN after attention
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_model),
        )
        self.ls_ffn = nn.Parameter(torch.tensor(1e-3))

        # ------------------------------------------------------------------
        # Output heads
        # ------------------------------------------------------------------
        self.head_raw = nn.Linear(d_model, d_model)

        # voxel-wise (x, y, z)
        self.head_voxel = nn.Linear(d_model, d_model * 3)

        # region-level connectivity (low-rank PSD)
        self.head_region_wide = nn.Linear(d_model, 256, bias=True)
        self.region_reduce = nn.Linear(256, rank, bias=False)

        self.scale_voxel = nn.Parameter(torch.zeros(1))
        self.scale_region = nn.Parameter(torch.zeros(1))

        self.register_buffer("eyeS", torch.eye(self.seq_out))

    def forward(self, z_low: torch.Tensor):
        """
        Args:
            z_low: (B, K_low, D)
                Low-level latent slots from the encoder.

        Returns:
            voxel  : (B, 21, 768, 3)
            region : (B, 21, 21)  PSD connectivity matrix
            raw    : (B, 21, 768)
        """
        B = z_low.size(0)

        # ------------------------------------------------------------------
        # Initialize grid tokens
        # ------------------------------------------------------------------
        kv = (
            self.grid_kv.expand(B, -1, -1)
            + self.coord_proj(self.grid_xy).unsqueeze(0)
        )
        h = kv  # (B, 21, D)

        # ------------------------------------------------------------------
        # Cross-attention decoding
        # ------------------------------------------------------------------
        for _ in range(self.layers):
            attn_out, _ = self.xattn(
                self.ln_q(h),
                self.ln_kv(z_low),
                self.ln_kv(z_low),
            )
            h = h + self.ls_dec * attn_out
            h = h + self.ls_ffn * self.ffn(h)

        # ------------------------------------------------------------------
        # Raw neural feature reconstruction
        # ------------------------------------------------------------------
        raw = self.head_raw(h)  # (B, 21, 768)

        # ------------------------------------------------------------------
        # Voxel topology reconstruction
        # ------------------------------------------------------------------
        voxel = self.head_voxel(h).view(B, self.seq_out, 768, 3)
        voxel = voxel * F.softplus(self.scale_voxel)

        # ------------------------------------------------------------------
        # Region connectivity (PSD)
        # ------------------------------------------------------------------
        Lw = self.head_region_wide(h)            # (B, 21, 256)
        L = self.region_reduce(Lw)               # (B, 21, rank)
        region = torch.bmm(L, L.transpose(1, 2)) / self.rank
        region = region * F.softplus(self.scale_region)
        region = region + 1e-4 * self.eyeS.to(region.dtype).unsqueeze(0)

        return voxel, region, raw
