import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==============================================================
# [MOD0] Utility: Projection Head & Fourier Positional Encoding
#        (freq bands now learnable)   <-- (item-6)
# ==============================================================
class FourierPositionalEncoding(nn.Module):
    """Learnable Fourier positional encoding (item-6)."""

    def __init__(self, num_bands: int = 6, d_model: int = 768):
        super().__init__()
        # Frequency bands are learnable parameters
        freq = 2.0 ** torch.linspace(0, num_bands - 1, num_bands)
        self.freq_bands = nn.Parameter(freq)  # (num_bands,)
        in_dim = 3 * num_bands * 2 * d_model
        self.proj = nn.Linear(in_dim, d_model, bias=False)

    def forward(self, xyz):
        """
        xyz: Input tensor with shape (B, 3, 21, 768), where:
        - B   : batch size
        - 3   : spatial coordinates (x, y, z)
        - 21  : number of brain regions
        - 768 : feature dimension per brain region
        """
        # Ensure input is a 4D tensor with shape (B, 3, 21, 768)
        if xyz.dim() == 4:
            B, _, num_regions, num_features = xyz.shape
        else:
            raise ValueError(
                f"Expected input shape (B, 3, 21, 768), but got {xyz.shape}"
            )

        # Apply Fourier encoding to (x, y, z) voxel coordinates
        xb = (2 * np.pi) * xyz.unsqueeze(-1) * self.freq_bands
        sin, cos = torch.sin(xb), torch.cos(xb)

        # Concatenate sine and cosine components
        feat = torch.cat([sin, cos], dim=-1)

        # Flatten the last two dimensions
        feat = feat.view(B, 3, num_regions, -1)

        # Flatten spatial dimensions (x, y, z) into feature vectors
        feat = feat.view(B, num_regions, -1)

        # Project features to the target output dimension (d_model)
        return self.proj(feat)  # Output shape: (B, 21, d_model)


# ------------------------------------------------------------------
# NEW --- Lightweight Voxel-Mixer
# ------------------------------------------------------------------
class VoxelMixer(nn.Module):
    """
    A minimal MLP-Mixer operating along both token and channel dimensions.
    Default depth=2 results in 4 sub-blocks (two token-mix and two channel-mix).
    Input shape: (B, T=21, C=768), fully aligned with the downstream Encoder.
    """
    def __init__(self, token_dim: int = 768, tokens: int = 21,
                 depth: int = 4, drop: float = 0.2):
        super().__init__()

        def _mlp(dim_in):
            return nn.Sequential(
                nn.Linear(dim_in, dim_in * 2),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(dim_in * 2, dim_in)
            )

        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            # Token-mixing (within-token channel mixing)
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(token_dim),
                _mlp(token_dim)
            ))
            # Channel-mixing (across-token mixing)
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(tokens),
                _mlp(tokens)
            ))

    def forward(self, x):                       # x: (B, T, C)
        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:                      # token-mix
                x = x + blk(x)
            else:                               # channel-mix
                x = x + blk(x.transpose(1, 2)).transpose(1, 2)
        return x                                # Output shape unchanged


# ==============================================================
# 1. Encoder — 21×768 → multi-level latents
#    (item-1 & item-2 & item-3)
# ==============================================================
class Encoder(nn.Module):
    def __init__(self,
                 d_model: int = 768,
                 num_heads: int = 4,
                 n_subj: int = 4,
                 depth_per_level: int = 2,
                 weight_decay: float = 1e-5):
        super().__init__()
        self.ntok = 21
        self.d_model = d_model
        self.n_low, self.n_mid1, self.n_mid2, self.n_high = 8, 4, 2, 2

        self.reg_pos_subj = nn.Parameter(torch.zeros(n_subj, self.ntok, d_model))
        self.fourier_pe = FourierPositionalEncoding(num_bands=6, d_model=d_model)

        self.film = nn.Sequential(
            nn.Linear(n_subj, 128), nn.GELU(),
            nn.Linear(128, d_model * 2)
        )

        self.pool_global = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )
        # Optional layer normalization for stability
        self.pool_norm = nn.LayerNorm(d_model)

        # VoxelMixer (added)
        self.senmantic = VoxelMixer(token_dim=self.d_model, tokens=self.ntok)

        # MLPs corresponding to each hierarchical level
        self.q_mlp_low  = nn.Linear(d_model, self.n_low  * d_model)
        self.q_mlp_mid1 = nn.Linear(d_model, self.n_mid1 * d_model)
        self.q_mlp_mid2 = nn.Linear(d_model, self.n_mid2 * d_model)
        self.q_mlp_high = nn.Linear(d_model, self.n_high * d_model)

        # Attention block (Multi-Head Attention)
        def _block():
            blk = nn.ModuleDict({
                "attn": nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1),
                "ln": nn.LayerNorm(d_model),  # Pre-Norm
            })
            # LayerScale: small residual coefficient for training stability
            blk.register_parameter("ls", nn.Parameter(1e-3 * torch.ones(self.d_model)))
            return blk

        self.b1 = nn.ModuleList([_block() for _ in range(depth_per_level)])
        self.b2 = nn.ModuleList([_block() for _ in range(depth_per_level)])
        self.b3 = nn.ModuleList([_block() for _ in range(depth_per_level)])
        self.b4 = nn.ModuleList([_block() for _ in range(depth_per_level)])

        # VAE heads for variational inference
        self.fc_mu_low   = nn.Linear(d_model, d_model)
        self.fc_lv_low   = nn.Linear(d_model, d_model)
        self.fc_mu_mid1  = nn.Linear(d_model, d_model)
        self.fc_lv_mid1  = nn.Linear(d_model, d_model)
        self.fc_mu_mid2  = nn.Linear(d_model, d_model)
        self.fc_lv_mid2  = nn.Linear(d_model, d_model)
        self.fc_mu_high  = nn.Linear(d_model, d_model)
        self.fc_lv_high  = nn.Linear(d_model, d_model)

        # Stable initialization: negative log-variance bias, zero weights
        for m in [self.fc_lv_low, self.fc_lv_mid1, self.fc_lv_mid2, self.fc_lv_high]:
            nn.init.constant_(m.bias, -2.0)
            nn.init.zeros_(m.weight)

        # Prior heads for each level
        self.prior_mu_mid1 = nn.Linear(d_model, d_model)
        self.prior_lv_mid1 = nn.Linear(d_model, d_model)
        self.prior_mu_mid2 = nn.Linear(d_model, d_model)
        self.prior_lv_mid2 = nn.Linear(d_model, d_model)
        self.prior_mu_low  = nn.Linear(d_model, d_model)
        self.prior_lv_low  = nn.Linear(d_model, d_model)
        self.prior_mu_high = nn.Linear(d_model, d_model)
        self.prior_lv_high = nn.Linear(d_model, d_model)

        # Residual scales for noise and prior regularization (kept small)
        self.ls_var_low = nn.Parameter(torch.tensor(1e-4))
        self.ls_var_mid1 = nn.Parameter(torch.tensor(1e-4))
        self.ls_var_mid2 = nn.Parameter(torch.tensor(1e-4))
        self.ls_var_high = nn.Parameter(torch.tensor(1e-4))

        self.ls_prior_low = nn.Parameter(torch.tensor(1e-4))
        self.ls_prior_mid1 = nn.Parameter(torch.tensor(1e-4))
        self.ls_prior_mid2 = nn.Parameter(torch.tensor(1e-4))
        self.ls_prior_high = nn.Parameter(torch.tensor(1e-4))

        # Small-scale posterior mean injection (ungated scheme A)
        self.ls_mu_low = nn.Parameter(torch.tensor(1e-3))
        self.ls_mu_mid1 = nn.Parameter(torch.tensor(1e-3))
        self.ls_mu_mid2 = nn.Parameter(torch.tensor(1e-3))
        self.ls_mu_high = nn.Parameter(torch.tensor(1e-3))

        # Training schedule parameters
        self.alpha_max   = 0.0
        self.beta_low    = 0.0
        self.beta_mid1   = 0.0
        self.beta_mid2   = 0.0
        self.beta_high   = 0.0
        self.detach_prior = False   # Dynamically controlled by training loop
        self.free_bits_tau = 0.0    # Set externally by training loop
        def _xattn(self, q, k, blk):
        y, _ = blk["attn"](blk["ln"](q), k, k)  # Q=ln(q), K=V=k
        return q + blk.ls * y

    # Replace _mix with the following implementation:
    # Inside Encoder, directly replace the original _mix
    def _mix(self, h, mu, lv, z_s, mu_p, ls_var, ls_prior, ls_mu=None):
        """
        Minimal variant: performs only small-scale posterior-mean injection.

        Args:
            h     : original slot representation (B, S, D)
            mu    : posterior mean (B, S, D)
            ls_mu : learnable scalar for each level (nn.Parameter),
                    initialized to a small value (1e-3)
        """
        # Use a small learnable scalar as a gate; tanh keeps the magnitude bounded
        s = torch.tanh(ls_mu) if ls_mu is not None else h.new_tensor(1e-3)
        return h + s * mu

    def _reparam(self, mu_layer, lv_layer, z):
        mu   = mu_layer(z)
        logv = lv_layer(z).clamp(-7.0, 1.0)          # Tighter range to avoid exp overflow
        eps  = torch.randn_like(mu)
        z_s  = mu + torch.exp(0.5 * logv) * eps
        return z_s, mu, logv

    def _kl_with_normal(self, mu, lv, tau: float = 0.0):
        var = lv.exp().clamp_min(1e-6)
        per = 0.5 * (mu.pow(2) + var - 1.0 - lv)  # (B, S, D)
        if tau > 0:
            per = torch.clamp(per - tau, min=0.0)
        return per.mean(dim=(1, 2))  # (B,)

    def _kl_diag_gauss(self, mu_q, lv_q, mu_p, lv_p, tau=0.0):
        var_q = lv_q.exp().clamp_min(1e-6)
        var_p = lv_p.exp().clamp_min(1e-6)
        per = 0.5 * (
            (lv_p - lv_q) +
            (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0
        )  # (B, S, D)
        if tau > 0:
            per = torch.clamp(per - tau, min=0.0)
        return per.mean(dim=(1, 2))  # (B,)

    def forward(self, x, subj_idx, xyz):

        # Semantic channel
        x_sem = self.senmantic(x)
        # x_sem = x

        # === Positional information injection (attenuated and switchable) ===
        pos = 0.1 * self.reg_pos_subj[subj_idx]  # (B, 21, D)
        pe = self.fourier_pe(xyz)                # (B, 21, D)
        x = x + pos + pe

        # === FiLM modulation (beta suppressed in early alignment stage) ===
        onehot = F.one_hot(
            subj_idx, num_classes=self.reg_pos_subj.size(0)
        ).float()
        gamma_raw, beta_raw = self.film(onehot).chunk(2, -1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)
        beta = 0.05 * torch.tanh(beta_raw)
        if getattr(self, "detach_prior", False):
            beta = beta * 0.0  # Disable beta shift in early training
        x = gamma.unsqueeze(1) * x + beta.unsqueeze(1)

        # Queries
        B = x.size(0)
        output, _  = self.pool_global(x)
        q_base = self.pool_norm(torch.mean(output, dim=1))

        q_low  = self.q_mlp_low(q_base).view(B, self.n_low,  self.d_model)
        q_mid1 = self.q_mlp_mid1(q_base).view(B, self.n_mid1, self.d_model)
        q_mid2 = self.q_mlp_mid2(q_base).view(B, self.n_mid2, self.d_model)
        q_high = self.q_mlp_high(q_base).view(B, self.n_high, self.d_model)

        def _stack(q, k, blocks):
            z = q
            for blk in blocks:
                z = self._xattn(z, k, blk)
            return z

        z_low  = _stack(q_low,  x_sem,                     self.b1)
        z_mid1 = _stack(q_mid1, torch.cat([z_low,  x_sem], 1), self.b2)
        z_mid2 = _stack(q_mid2, torch.cat([z_mid1, x_sem], 1), self.b3)
        z_high = _stack(q_high, torch.cat([z_mid2, x_sem], 1), self.b4)

        # Reparameterization
        z_low_s,  mu_l,  lv_l  = self._reparam(self.fc_mu_low,  self.fc_lv_low,  z_low)
        z_mid_s1, mu_m1, lv_m1 = self._reparam(self.fc_mu_mid1, self.fc_lv_mid1, z_mid1)
        z_mid_s2, mu_m2, lv_m2 = self._reparam(self.fc_mu_mid2, self.fc_lv_mid2, z_mid2)
        z_high_s, mu_h,  lv_h  = self._reparam(self.fc_mu_high, self.fc_lv_high, z_high)

        # Parent representations (means used as parents; detachment controlled by training loop)
        parent_high = mu_h.mean(dim=1)
        parent_mid1 = mu_m1.mean(dim=1)
        parent_mid2 = mu_m2.mean(dim=1)
        if self.detach_prior:
            parent_high = parent_high.detach()
            parent_mid1 = parent_mid1.detach()
            parent_mid2 = parent_mid2.detach()

        # Conditional priors (numerically stable; log-variance clipped to [-7, 1])
        mu_p_mid1 = self.prior_mu_mid1(parent_high)[:, None, :].expand_as(mu_m1)
        lv_p_mid1 = self.prior_lv_mid1(parent_high)[:, None, :].expand_as(lv_m1).clamp(-7.0, 1.0)

        mu_p_mid2 = self.prior_mu_mid2(parent_mid1)[:, None, :].expand_as(mu_m2)
        lv_p_mid2 = self.prior_lv_mid2(parent_mid1)[:, None, :].expand_as(lv_m2).clamp(-7.0, 1.0)

        mu_p_low  = self.prior_mu_low(parent_mid2)[:, None, :].expand_as(mu_l)
        lv_p_low  = self.prior_lv_low(parent_mid2)[:, None, :].expand_as(lv_l).clamp(-7.0, 1.0)

        tau = getattr(self, "free_bits_tau", 0.0)

        # KL divergence:
        # high level vs. standard normal, lower levels vs. conditional priors
        kl_high = self._kl_with_normal(mu_h,  lv_h,  tau)
        kl_mid1 = self._kl_diag_gauss(mu_m1, lv_m1, mu_p_mid1, lv_p_mid1, tau)
        kl_mid2 = self._kl_diag_gauss(mu_m2, lv_m2, mu_p_mid2, lv_p_mid2, tau)
        kl_low  = self._kl_diag_gauss(mu_l,  lv_l,  mu_p_low,  lv_p_low,  tau)

        kl = (kl_low + kl_mid1 + kl_mid2 + kl_high).mean()

        # Residual injection with posterior mean
        z_low_out = self._mix(
            z_low, mu_l, lv_l, z_low_s, mu_p_low,
            self.ls_var_low, self.ls_prior_low, self.ls_mu_low
        )

        z_mid1_out = self._mix(
            z_mid1, mu_m1, lv_m1, z_mid_s1, mu_p_mid1,
            self.ls_var_mid1, self.ls_prior_mid1, self.ls_mu_mid1
        )

        z_mid2_out = self._mix(
            z_mid2, mu_m2, lv_m2, z_mid_s2, mu_p_mid2,
            self.ls_var_mid2, self.ls_prior_mid2, self.ls_mu_mid2
        )

        z_high_out = self._mix(
            z_high, mu_h, lv_h, z_high_s, torch.zeros_like(mu_h),
            self.ls_var_high, self.ls_prior_high, self.ls_mu_high
        )

        z_out = {
            "z_low":  z_low_out,
            "z_mid1": z_mid1_out,
            "z_mid2": z_mid2_out,
            "z_high": z_high_out,
        }

        # Return posterior means as stable representations for downstream mapper
        z_low_s  = mu_l
        z_mid_s1 = mu_m1
        z_mid_s2 = mu_m2
        z_high_s = mu_h

        return z_low_s, z_mid_s1, z_mid_s2, z_high_s, kl, z_out, x_sem

