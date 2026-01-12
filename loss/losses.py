"""
Loss functions for hierarchical brain–semantic alignment.

This file provides a reference implementation of the loss components
used in HTA-NRM, including:
- Safe Sinkhorn (optimal transport) loss
- Wasserstein-Infused Contrastive (WIC) loss
- Mutual information estimator (InfoNCE)

Note:
This implementation is released for research and review purposes.
Due to stochastic components (e.g., Sinkhorn stabilization, hard-negative
sampling, and numerical safeguards), exact numerical reproducibility is
not guaranteed at this stage.

A fully reproducible and ablation-complete version will be released
upon paper acceptance.
"""


from __future__ import annotations

# ------------------------
# Standard library
# ------------------------
from typing import Dict

# ------------------------
# PyTorch
# ------------------------
import torch
import torch.nn.functional as F


# ==============================================================
# Sinkhorn / Optimal Transport
# ==============================================================

def safe_sinkhorn_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    blur: float = 0.03,
    scaling: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Numerically safe Sinkhorn (OT) loss.

    Args:
        x, y:
            Input tensors of shape (B, N, D), (B, D), or (N, D).
        blur:
            Sinkhorn blur parameter.
        scaling:
            Sinkhorn scaling parameter in (0, 1).
        eps:
            Small value for variance stabilization.

    Returns:
        A scalar loss tensor.
    """

    # Normalize shapes to (B, N, D)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)

    assert x.dim() == 3 and y.dim() == 3, \
        f"x/y must be (B,N,D), got {x.shape}, {y.shape}"
    assert 0.0 < scaling < 1.0, f"scaling must be in (0,1), got {scaling}"
    assert blur > 0.0, f"blur must be > 0, got {blur}"

    x = x.contiguous()
    y = y.contiguous()

    # L2 normalization
    x = F.normalize(x, dim=-1, eps=1e-12)
    y = F.normalize(y, dim=-1, eps=1e-12)

    # Sanitize NaN / Inf
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

    # Variance safeguard
    if x.std() < eps:
        x = x + eps * torch.randn_like(x)
    if y.std() < eps:
        y = y + eps * torch.randn_like(y)

    # Call geomloss explicitly in FP32 (keep gradients)
    try:
        import geomloss

        # Explicitly disable autocast for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            loss_fn = geomloss.SamplesLoss(
                loss="sinkhorn",
                p=2,
                blur=float(blur),
                scaling=float(scaling),
            )
            loss = loss_fn(
                x.to(torch.float32),
                y.to(torch.float32),
            )
        return loss

    except Exception as e:
        # Fallback: MSE (still differentiable)
        print("[warn] Sinkhorn failed, fallback to MSE:", repr(e))
        return torch.mean((x - y) ** 2)


# ==============================================================
# WIC: Wasserstein-Infused Contrastive Loss
# ==============================================================

def WIC_Loss(
    preds: torch.Tensor,
    targs: torch.Tensor,
    stimuli: torch.Tensor,
    *,
    temp: float = 0.05,
    eps: float = 1e-12,
    margin: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Wasserstein-Infused Contrastive (WIC) loss.

    Args:
        preds:
            Predicted embeddings, shape (B, ...).
        targs:
            Target embeddings, shape (B, ...).
        stimuli:
            Stimulus IDs for hard-negative filtering, shape (B,).
        temp:
            Temperature for contrastive logits.
        eps:
            Small value for normalization.
        margin:
            Margin for hard-negative hinge loss.

    Returns:
        Dictionary containing individual loss terms and total loss.
    """

    B = preds.shape[0]
    preds = preds.reshape(B, -1).contiguous()
    targs = targs.reshape(B, -1).contiguous()

    preds = F.normalize(preds, p=2, dim=-1, eps=eps)
    targs = F.normalize(targs, p=2, dim=-1, eps=eps)

    temp_t = torch.as_tensor(temp, dtype=preds.dtype, device=preds.device)

    # Similarity logits (centered for stability)
    clip_clip = (targs @ targs.T) / temp_t
    brain_clip = (preds @ targs.T) / temp_t
    clip_clip = clip_clip - clip_clip.max(dim=-1, keepdim=True).values
    brain_clip = brain_clip - brain_clip.max(dim=-1, keepdim=True).values

    # Soft-label KD / symmetric CE
    prob_tt = clip_clip.softmax(dim=-1)
    loss1 = -(F.log_softmax(brain_clip, dim=-1) * prob_tt).sum(-1).mean()
    loss2 = -(F.log_softmax(brain_clip.T, dim=-1) * prob_tt.T).sum(-1).mean()
    contrastive_loss = 0.5 * (loss1 + loss2)

    # Hard negatives (skip if B == 1)
    if B > 1:
        same_stim_mask = (stimuli.unsqueeze(1) == stimuli.unsqueeze(0))
        diag_mask = torch.eye(B, dtype=torch.bool, device=preds.device)
        bad_mask = same_stim_mask | diag_mask

        neg = brain_clip.masked_fill(bad_mask, float("-inf"))
        k = min(8, B - 1)
        hard_vals, _ = torch.topk(neg, k=k, dim=1)

        pos = torch.diag(brain_clip)
        hinge = F.relu(margin / temp_t + hard_vals - pos.unsqueeze(1))

        with torch.no_grad():
            weights = hard_vals.detach().softmax(dim=1)

        hard_negative_loss = (hinge * weights).sum(dim=1).mean()
    else:
        hard_negative_loss = preds.new_zeros(())

    # Wasserstein / OT
    wasserstein_loss = safe_sinkhorn_loss(preds, targs)

    # Variance regularization
    variance_loss = -(
        preds.var(dim=1, unbiased=False).mean()
        + targs.var(dim=1, unbiased=False).mean()
    )

    total_loss = (
        contrastive_loss
        + wasserstein_loss
        + hard_negative_loss
        + variance_loss
    )

    return {
        "contrastive_loss": contrastive_loss,
        "hard_negative_loss": hard_negative_loss,
        "wasserstein_loss": wasserstein_loss,
        "variance_loss": variance_loss,
        "total_loss": total_loss,
    }


# ==============================================================
# Mutual Information (InfoNCE)
# ==============================================================

def compute_mutual_information(
    x: torch.Tensor,
    y: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    Mutual information estimator using symmetric InfoNCE.

    Args:
        x, y:
            Feature tensors of shape (B, D) on the same device.
        temperature:
            Softmax temperature.

    Returns:
        Scalar InfoNCE loss (lower is better).
    """

    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    logits = torch.matmul(x, y.T) / temperature
    labels = torch.arange(x.size(0), device=x.device)

    loss_xy = F.cross_entropy(logits, labels)
    loss_yx = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_xy + loss_yx)
