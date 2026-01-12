import torch
import torch.nn.functional as F

# ── ④ Shape handling & normalization before Sinkhorn; fallback for extreme cases ──
def safe_sinkhorn_loss(x, y, *, blur=0.03, scaling=0.5, eps=1e-6):
    """
    x, y: (B, N, D) or (B, D) or (N, D)
    - Call geomloss in FP32 with autocast disabled while preserving gradients
    - Normalize inputs to (B, N, D), clean NaN/Inf, add small noise for near-zero variance
    - Returns a scalar loss that can be directly added to the total objective
    """
    # Normalize input shapes to (B, N, D)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
    assert x.dim() == 3 and y.dim() == 3, \
        f"x/y shape must be (B,N,D), got {x.shape}, {y.shape}"
    assert 0 < scaling < 1, f"scaling must be in (0,1), got {scaling}"
    assert blur > 0, f"blur must be > 0, got {blur}"

    # Ensure contiguous memory
    x = x.contiguous()
    y = y.contiguous()

    # L2 normalization with explicit epsilon
    x = F.normalize(x, dim=-1, eps=1e-12)
    y = F.normalize(y, dim=-1, eps=1e-12)

    # Sanitize NaN / Inf values
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

    # Inject small noise if variance is extremely small
    if x.std() < eps:
        x = x + eps * torch.randn_like(x)
    if y.std() < eps:
        y = y + eps * torch.randn_like(y)

    # Call GeomLoss in FP32 with autocast disabled (keep gradients)
    try:
        import geomloss
        with torch.autocast(device_type="cuda"):
            loss_fn = geomloss.SamplesLoss(
                loss="sinkhorn", p=2, blur=float(blur), scaling=float(scaling)
            )
            loss = loss_fn(x.to(torch.float32), y.to(torch.float32))
        return loss  # scalar
    except Exception as e:
        # Print diagnostic message once, then fall back to MSE (still differentiable)
        print("[warn] sinkhorn failed, fallback to MSE:", repr(e))
        return torch.mean((x - y) ** 2)


def WIC_loss(preds, targs, stimuli,
                                     temp=0.05, eps=1e-12, margin=0.2):
    B = preds.shape[0]
    preds = preds.reshape(B, -1).contiguous()
    targs = targs.reshape(B, -1).contiguous()

    # L2 normalization with explicit epsilon
    preds = F.normalize(preds, p=2, dim=-1, eps=eps)
    targs = F.normalize(targs, p=2, dim=-1, eps=eps)

    temp_t = torch.as_tensor(temp, dtype=preds.dtype, device=preds.device)

    # Logits (centered for numerical stability)
    clip_clip  = (targs @ targs.T) / temp_t
    brain_clip = (preds @ targs.T) / temp_t
    clip_clip  = clip_clip  - clip_clip.max(dim=-1, keepdim=True).values
    brain_clip = brain_clip - brain_clip.max(dim=-1, keepdim=True).values

    # Soft-label knowledge distillation / symmetric cross-entropy
    prob_tt = clip_clip.softmax(-1)
    loss1 = -(F.log_softmax(brain_clip,  dim=-1) * prob_tt).sum(-1).mean()
    loss2 = -(F.log_softmax(brain_clip.T, dim=-1) * prob_tt.T).sum(-1).mean()
    contrastive_loss = 0.5 * (loss1 + loss2)

    # Hard negatives (zero if B == 1)
    # Hard negatives with same-stimulus filtering
    if B > 1:
        stim_ids = stimuli
        same_stim_mask = (stim_ids.unsqueeze(1) == stim_ids.unsqueeze(0))  # (B,B)

        # Mask diagonal (positive pairs)
        diag_mask = torch.eye(B, dtype=torch.bool, device=brain_clip.device)

        # Combine masks: same stimulus OR self-pair
        bad_mask = diag_mask | same_stim_mask

        # Exclude invalid negatives by setting to -inf
        neg = brain_clip.masked_fill(bad_mask, float("-inf"))

        k = min(8, B - 1)
        hard_vals, _ = torch.topk(neg, k=k, dim=1)  # (B,k)

        pos = torch.diag(brain_clip)

        margin_scaled = margin / temp_t
        hinge = F.relu(margin_scaled + hard_vals - pos.unsqueeze(1))

        with torch.no_grad():
            weights = hard_vals.detach().softmax(dim=1)
        hard_negative_loss = (hinge * weights).sum(dim=1).mean()
    else:
        hard_negative_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)

    # Sinkhorn / OT loss: safe_sinkhorn_loss already returns a scalar
    wasserstein_loss = safe_sinkhorn_loss(preds, targs)

    # Variance regularization (encourage diversity)
    variance_loss = -(
        preds.var(dim=1, unbiased=False).mean() +
        targs.var(dim=1, unbiased=False).mean()
    )

    total_loss = (
        contrastive_loss +
        wasserstein_loss +
        hard_negative_loss +
        variance_loss
    )

    return {
        "contrastive_loss": contrastive_loss,
        "hard_negative_loss": hard_negative_loss,
        "wasserstein_loss": wasserstein_loss,
        "variance_loss": variance_loss,
        "total_loss": total_loss,
    }


def compute_mutual_information(x: torch.Tensor,
                               y: torch.Tensor,
                               temperature: float = 0.2) -> torch.Tensor:
    """
    x, y : (B, D) on the same device
    Returns the InfoNCE loss (lower is better), which serves as an upper bound on MI.
    """
    # 1) L2 normalization to ensure valid cosine similarity
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    # 2) Similarity matrix (B, B)
    logits = torch.matmul(x, y.t()) / temperature  # similarities between x_i and all y_j

    # 3) i == j are positives, others are negatives — InfoNCE
    labels = torch.arange(x.size(0), device=x.device)

    loss_xy = F.cross_entropy(logits, labels)       # x as query, y as key
    loss_yx = F.cross_entropy(logits.t(), labels)   # y as query, x as key

    return 0.5 * (loss_xy + loss_yx)                # symmetric form
