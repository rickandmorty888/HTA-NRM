import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from encoder import Encoder 
from generator import Generator


class ProjectionHead(nn.Module):
    """2-layer MLP head used in contrastive learning (SimCLR / CLIP style)."""
    def __init__(self, dim_in: int = 768, dim_hidden: int = 1024,
                 dim_out: int = 768, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(dim_hidden, dim_out)  # Output dimension matches projection space
        )
        self.ln = nn.LayerNorm(dim_out)  # LayerNorm after projection

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, N, D),
               where B is batch size, N is token count, and D is feature dimension.

        Returns:
            Projected tensor of shape (B, N, D).
        """
        B, N, D = x.shape
        x = x.view(B, -1)  # Flatten across token dimension
        x = self.net(x)
        x = self.ln(x)
        return x


# ---------------------------------------------------------------------------
# 3. Mapper (retains ProjectionHead upgrades)
# ---------------------------------------------------------------------------
class MapperWithClassification(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 768, rank: int = 16):
        super().__init__()

        # Projection heads for contrastive alignment
        self.proj_dino = ProjectionHead(dim_in=d_model * 8, dim_hidden=1024, dim_out=d_model)
        self.proj_clip_img = ProjectionHead(dim_in=d_model * 4, dim_hidden=1024, dim_out=d_model)
        self.proj_clip_txt = ProjectionHead(dim_in=d_model * 2, dim_hidden=1024, dim_out=d_model)
        self.proj_llama = ProjectionHead(
            dim_in=d_model * 2, dim_hidden=1024, dim_out=4096
        )  # 4096-dim output for LLaMA features

        # Classification heads
        self.classifier_dino = nn.Linear(d_model, num_classes)
        self.classifier_clip_img = nn.Linear(d_model, num_classes)
        self.classifier_clip_txt = nn.Linear(d_model, num_classes)
        self.classifier_llama = nn.Linear(4096, num_classes)

    def forward(self, z_low, z_mid1, z_mid2, z_high,
                mask_low=None, mask_mid1=None, mask_mid2=None, mask_high=None):

        # Obtain projected representations for each hierarchy level
        d_raw = self.proj_dino(z_low)
        i_raw = self.proj_clip_img(z_mid1)
        t_raw = self.proj_clip_txt(z_mid2)
        l_raw = self.proj_llama(z_high)

        # Normalization for alignment
        d = F.normalize(d_raw, dim=-1)
        i = F.normalize(i_raw, dim=-1)
        t = F.normalize(t_raw, dim=-1)
        l = F.normalize(l_raw, dim=-1)

        # Classification for each latent representation
        d_class = self.classifier_dino(d)
        i_class = self.classifier_clip_img(i)
        t_class = self.classifier_clip_txt(t)
        l_class = self.classifier_llama(l)

        # Returns:
        # 1) normalized embeddings for alignment
        # 2) raw embeddings for VICReg / CORAL losses
        # 3) classification logits
        return (d, i, t, l), (d_raw, i_raw, t_raw, l_raw), (d_class, i_class, t_class, l_class)


# ==============================================================
# 4. MAE — integrates encoder / generator with beta annealing
# ==============================================================
class MAE(nn.Module):
    """
    Masked Autoencoder (MAE) with token-level random masking.

    Inputs (batch dictionary):
        - RawBeta    : (B, N, D), e.g., (B, 21, 768)
        - Voxel_Topo : (B, N, K >= 3), spatial coordinates (xyz = [..., :3])
                       used only by the encoder
        - subj_idx   : (B,)

    Returns:
        kl, voxel, region, raw,
        latent_dino, latent_clip_img, latent_clip_txt, latent_llama,
        token_mask
    """

    def __init__(
        self,
        mask_rate: float = 0.0,      # Initial masking ratio
        max_mask_rate: float = 0.4,  # Maximum masking ratio at later epochs
        total_epochs: int = 50,
        n_subj: int = 4,
        num_classes: int = 80,
        use_learned_mask_token: bool = True,
        same_mask_across_batch: bool = False,  # Whether all samples share the same mask
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(n_subj=n_subj)
        self.generator = Generator()
        self.mapper = MapperWithClassification(num_classes=num_classes)
        self.classification_loss_fn = nn.BCEWithLogitsLoss()

        self.mask_rate = mask_rate
        self.max_mask = max_mask_rate
        self.total_ep = total_epochs
        self.use_learned = use_learned_mask_token
        self.same_across = same_mask_across_batch

        # Temperature / margin parameters (kept from original design)
        self.logit_scale_clip = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_dino = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_sem1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_sem2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.margin = nn.Parameter(torch.tensor(0.2))

        # Learnable [MASK] token (more stable than zero masking)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 768))

        # Task uncertainty weights (four tasks: dino, clip, sem1, sem2)
        self.log_vars = nn.ParameterDict({
            "dino": nn.Parameter(torch.zeros(1)),
            "clip": nn.Parameter(torch.zeros(1)),
            "sem1": nn.Parameter(torch.zeros(1)),
            "sem2": nn.Parameter(torch.zeros(1)),
        })

        # Average pooling over the token dimension (N = 21)
        self.avgpool = nn.AvgPool1d(kernel_size=21, stride=1)

    # -------- masking ratio schedule --------
    def _mask_ratio(self, epoch: int, train: bool, override: float = None):
        if override is not None:
            return float(override)
        if not train:
            return 0.0
        t = max(0.0, min(epoch / self.total_ep, 1.0))
        return float(self.mask_rate + (self.max_mask - self.mask_rate) * t)

    # -------- number of masked tokens --------
    @staticmethod
    def _num_mask_tokens(N: int, r: float, train: bool) -> int:
        if (not train) or r <= 0.0:
            return 0
        k = int(round(N * r))
        # Avoid masking all or none of the tokens during training
        k = max(1, min(N - 1, k)) if N >= 2 else 0
        return k

    # -------- random token masking --------
    def _mask_tokens(self, x: torch.Tensor, r: float, train: bool):
        """
        Args:
            x: Input tensor of shape (B, N, D)

        Returns:
            x_masked: Masked input tensor (B, N, D)
            m       : Boolean mask (B, N), True indicates masked tokens
        """
        B, N, D = x.shape
        k = self._num_mask_tokens(N, r, train)

        if k == 0:
            return x, torch.zeros(B, N, dtype=torch.bool, device=x.device)

        if self.same_across:
            idx = torch.randperm(N, device=x.device)[:k]
            m = torch.zeros(N, device=x.device, dtype=torch.bool)
            m[idx] = True
            m = m.unsqueeze(0).expand(B, N)
        else:
            m = torch.zeros(B, N, device=x.device, dtype=torch.bool)
            for b in range(B):
                idx = torch.randperm(N, device=x.device)[:k]
                m[b, idx] = True

        x_masked = x.clone()
        if self.use_learned:
            x_masked[m] = self.mask_token.expand(B, N, D)[m]
        else:
            x_masked[m] = 0.0
        return x_masked, m

    # -------- forward --------
    def forward(self, batch: dict, epoch: int,
                mask_ratio: float = None, train: bool = True):
        """
        Returns:
            kl, voxel, region, raw,
            latent_dino, latent_clip_img, latent_clip_txt, latent_llama,
            token_mask, classification_loss, pooled_semantic
        """
        x = batch["RawBeta"]            # (B, N, D)
        xyz = batch["Voxel_Topo"][...]  # (B, N, 3), passed to encoder only
        subj_idx = batch["subj_idx"].long()

        # Compute masking ratio for the current epoch
        r = self._mask_ratio(epoch, train, mask_ratio)

        # Apply random token masking
        x_mask, token_mask = self._mask_tokens(x, r, train)

        # Encoder
        z_low_s, z_mid_s1, z_mid_s2, z_high_s, kl, z_out, x_sem = \
            self.encoder(x_mask, subj_idx, xyz)

        # Mapper (with classification heads)
        (z_dino, z_clip_img, z_clip_txt, z_llama), \
        (z_dino_raw, z_clip_img_raw, z_clip_txt_raw, z_llama_raw), \
        (d_class, i_class, t_class, l_class) = \
            self.mapper(z_low_s, z_mid_s1, z_mid_s2, z_high_s)

        # Generator
        voxel, region, raw = self.generator(z_out["z_low"])

        # Classification loss
        labels = batch["label_80"].to(self.device)
        class_loss = (
            self.classification_loss_fn(d_class, labels) +
            self.classification_loss_fn(i_class, labels) +
            self.classification_loss_fn(t_class, labels) +
            self.classification_loss_fn(l_class, labels)
        )

        # Pool semantic features across tokens
        input_tensor = x_sem.permute(0, 2, 1)  # (B, 768, 21)
        output_tensor = self.avgpool(input_tensor).squeeze(-1)  # (B, 768)

        return (
            kl, voxel, region, raw,
            z_dino, z_clip_img, z_clip_txt, z_llama,
            z_dino_raw, z_clip_img_raw, z_clip_txt_raw, z_llama_raw,
            token_mask, class_loss, output_tensor
        )
