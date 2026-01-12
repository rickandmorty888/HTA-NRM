# ==============================================================
# Alignment-only Training with VICReg + CORAL
# Uses WIC_Loss for semantic alignment
# ==============================================================

from __future__ import annotations

# ------------------------
# Standard library
# ------------------------
import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional

# ------------------------
# PyTorch
# ------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------
# Project imports
# ------------------------
from models import MAE
from dataset import BrainDataset
from losses import WIC_Loss, compute_mutual_information


# ==============================================================
# Utils
# ==============================================================

def set_seed(seed: Optional[int], deterministic: bool = False) -> None:
    """Set random seeds (optional) and determinism flags."""
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism can reduce performance; keep it optional.
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # This may raise errors for some ops; enable only if you really want it:
        # torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def cosine01(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 * (1.0 - math.cos(math.pi * t))


def save_checkpoint(
    ckpt_dir: str,
    name: str,
    payload: Dict[str, Any],
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, name)
    torch.save(payload, path)
    return path


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not files:
        return None

    # Prefer "last.pth" if present
    if "last.pth" in files:
        return os.path.join(ckpt_dir, "last.pth")

    # Else pick newest by mtime
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)))
    return os.path.join(ckpt_dir, files[-1])


# ==============================================================
# Argument parser
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Align-only training with VICReg + CORAL")

    # -------- Paths --------
    p.add_argument("--data_root", type=str, default="data", help="Root directory of dataset")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/align_only", help="Directory to save checkpoints")
    p.add_argument("--run_name", type=str, default="", help="Optional subfolder name under ckpt_dir")

    # -------- Resume --------
    # resume: "auto" -> latest in ckpt_dir; "none" -> no resume; or provide a .pth path
    p.add_argument("--resume", type=str, default="none", help='Resume mode: "auto", "none", or /path/to/ckpt.pth')

    # -------- Training --------
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true", help="Use pin_memory for DataLoader")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--print_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1, help="Save epoch checkpoint every N epochs")

    # -------- AMP --------
    p.add_argument("--amp", action="store_true", help="Enable torch autocast + GradScaler (CUDA only)")

    # -------- Seed / Determinism --------
    p.add_argument("--seed", type=int, default=None, help="Random seed (default: None = not set)")
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic CuDNN behavior")

    # -------- Masking schedule --------
    p.add_argument("--mask_min", type=float, default=0.05)
    p.add_argument("--mask_max", type=float, default=0.20)

    # -------- Loss weights (global) --------
    p.add_argument("--lambda_vic", type=float, default=0.05)
    p.add_argument("--lambda_coral", type=float, default=0.05)
    p.add_argument("--lambda_cls", type=float, default=1.0)
    p.add_argument("--lambda_sem", type=float, default=1.0)
    p.add_argument("--lambda_kl", type=float, default=0.01)

    # MAE-related aux terms
    p.add_argument("--lambda_rec", type=float, default=0.05)
    p.add_argument("--lambda_mi", type=float, default=0.03)

    # Align term scale
    p.add_argument("--lambda_align", type=float, default=2.0)

    # -------- WIC internal weights (keep your defaults) --------
    p.add_argument("--wic_contrastive_w", type=float, default=2.0)
    p.add_argument("--wic_wasserstein_w", type=float, default=0.30)
    # hard_negative and variance are already included as-is in your original expression

    # -------- Temperatures --------
    p.add_argument("--temp_clip", type=float, default=0.03)
    p.add_argument("--temp_dino", type=float, default=0.03)
    p.add_argument("--temp_sem", type=float, default=0.03)
    p.add_argument("--temp_text", type=float, default=0.05)
    p.add_argument("--temp_llama", type=float, default=0.05)

    return p.parse_args()


# ==============================================================
# Loss helpers
# ==============================================================

def vicreg_loss(z1, z2, sim_w=1.0, std_w=1.0, cov_w=0.04, eps=1e-4):
    """VICReg: invariance + variance + covariance regularization."""
    sim_loss = F.mse_loss(z1, z2)

    def _std(z):
        z = z - z.mean(0, keepdim=True)
        return torch.mean(F.relu(1.0 - torch.sqrt(z.var(0, unbiased=False) + eps)))

    def _cov(z):
        z = z - z.mean(0, keepdim=True)
        cov = (z.T @ z) / max(1, z.size(0) - 1)
        off = cov - torch.diag(torch.diag(cov))
        return (off ** 2).sum() / z.size(1)

    return sim_w * sim_loss + std_w * (_std(z1) + _std(z2)) + cov_w * (_cov(z1) + _cov(z2))


def coral_loss(z, domain_ids):
    """CORAL across subjects: match covariance between domains."""
    uniq = domain_ids.unique()
    covs = []
    for u in uniq:
        zu = z[domain_ids == u]
        if zu.size(0) < 2:
            continue
        zu = zu - zu.mean(0, keepdim=True)
        covs.append((zu.T @ zu) / max(1, zu.size(0) - 1))
    if len(covs) < 2:
        return z.new_zeros(())
    loss, cnt = 0.0, 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            loss += F.mse_loss(covs[i], covs[j])
            cnt += 1
    return loss / max(1, cnt)


def wic_sloss(pred, targ, stimuli, temp, margin, contrastive_w=2.0, wasserstein_w=0.30):
    """Your original _sloss formula, with weights exposed."""
    d = WIC_Loss(pred, targ, stimuli, temp=temp, margin=margin)
    return (
        contrastive_w * d["contrastive_loss"]
        + d["hard_negative_loss"]
        + wasserstein_w * d["wasserstein_loss"]
        + d["variance_loss"]
    ), d


# ==============================================================
# Main
# ==============================================================

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    set_seed(args.seed, deterministic=args.deterministic)

    ckpt_dir = args.ckpt_dir
    if args.run_name:
        ckpt_dir = os.path.join(ckpt_dir, args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ======================
    # Dataset configuration
    # ======================
    subject_ids = ["subj01", "subj02", "subj05", "subj07"]

    brain_file_paths = [
        os.path.join(args.data_root, sid, "p2", f"concatenated_brain_data_{sid}.h5py")
        for sid in subject_ids
    ]
    excel_file_paths = [
        os.path.join(args.data_root, sid, f"trailID-{sid}.csv")
        for sid in subject_ids
    ]
    indices_sorted = list(range(21))

    train_dataset = BrainDataset(
        subject_ids=subject_ids,
        brain_file_paths=brain_file_paths,
        excel_file_paths=excel_file_paths,
        indices_sorted=indices_sorted,
        is_train=True,
        valid_ratio=0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    steps_per_epoch = len(train_loader)

    # ======================
    # Model
    # ======================
    mae_model = MAE(
        mask_rate=0.0,
        max_mask_rate=args.mask_max,
        total_epochs=args.epochs,
        n_subj=len(subject_ids)
    ).to(device)

    # Fix temperature parameters on the model (keep your behavior)
    with torch.no_grad():
        for val, name in [
            (args.temp_clip, "logit_scale_clip"),
            (args.temp_dino, "logit_scale_dino"),
            (args.temp_sem,  "logit_scale_sem1"),
            (args.temp_text, "logit_scale_sem2"),
        ]:
            getattr(mae_model, name).copy_(torch.tensor(math.log(1.0 / float(val)), device=device))
            getattr(mae_model, name).requires_grad = False

    # ======================
    # Optimizer & scheduler
    # ======================
    optimizer = torch.optim.AdamW(
        [p for p in mae_model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(1, args.epochs * max(1, steps_per_epoch)),
        pct_start=0.1
    )

    criterion = nn.MSELoss()

    # ======================
    # Resume
    # ======================
    start_epoch = 0
    global_step = 0

    if args.resume != "none":
        if args.resume == "auto":
            resume_path = find_latest_checkpoint(ckpt_dir)
        else:
            resume_path = args.resume

        if resume_path and os.path.isfile(resume_path):
            print(f"[resume] Loading checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            mae_model.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("global_step", 0))
            print(f"[resume] start_epoch={start_epoch}, global_step={global_step}")
        else:
            print(f"[resume] Not found: {resume_path}. Training from scratch.")

    # ======================
    # Training loop
    # ======================
    for epoch in range(start_epoch, args.epochs):
        mae_model.train()
        t0 = time.time()

        for it, batch in enumerate(train_loader):
            global_step += 1

            # Move tensors to device
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            # --- two-view masking schedule ---
            phase = cosine01((epoch * steps_per_epoch + it + 1) / max(1, args.epochs * steps_per_epoch))
            mask1 = args.mask_min + (args.mask_max - args.mask_min) * phase
            mask2 = max(args.mask_min, mask1 * (0.9 + 0.2 * float(torch.rand((), device=device).item())))

            with torch.cuda.amp.autocast(enabled=use_amp):
                out1 = mae_model(batch, epoch, mask1, train=True)
                out2 = mae_model(batch, epoch, mask2, train=True)

                (
                    kl1, voxel1, region1, raw1,
                    d1, i1, t1, l1,
                    d1_raw, i1_raw, t1_raw, l1_raw,
                    _, class_loss1, z_sem1
                ) = out1

                (
                    kl2, voxel2, region2, raw2,
                    d2, i2, t2, l2,
                    d2_raw, i2_raw, t2_raw, l2_raw,
                    _, class_loss2, z_sem2
                ) = out2

                # ---- alignment losses (WIC) ----
                L_align = 0.0
                wic_logs = {}

                ld, dd = wic_sloss(d1, batch["dino_Img"], batch["stimuli"], args.temp_dino, mae_model.margin,
                                   contrastive_w=args.wic_contrastive_w, wasserstein_w=args.wic_wasserstein_w)
                li, di = wic_sloss(i1, batch["CLIP_embedding"], batch["stimuli"], args.temp_clip, mae_model.margin,
                                   contrastive_w=args.wic_contrastive_w, wasserstein_w=args.wic_wasserstein_w)
                lt, dt = wic_sloss(t1, batch["clip_cap"], batch["stimuli"], args.temp_text, mae_model.margin,
                                   contrastive_w=args.wic_contrastive_w, wasserstein_w=args.wic_wasserstein_w)
                ll, dl = wic_sloss(l1, batch["llama_cap"], batch["stimuli"], args.temp_llama, mae_model.margin,
                                   contrastive_w=args.wic_contrastive_w, wasserstein_w=args.wic_wasserstein_w)

                L_align = ld + li + lt + ll
                wic_logs["wic_dino_total"] = ld
                wic_logs["wic_clip_total"] = li
                wic_logs["wic_text_total"] = lt
                wic_logs["wic_llama_total"] = ll

                # Semantic alignment (semantic pooled output to CLIP)
                L_sem, dsem = wic_sloss(
                    z_sem1, batch["CLIP_embedding"], batch["stimuli"],
                    args.temp_sem, mae_model.margin,
                    contrastive_w=args.wic_contrastive_w,
                    wasserstein_w=args.wic_wasserstein_w
                )

                # VICReg (raw)
                L_vic = (
                    vicreg_loss(d1_raw, d2_raw) +
                    vicreg_loss(i1_raw, i2_raw) +
                    vicreg_loss(t1_raw, t2_raw) +
                    vicreg_loss(l1_raw, l2_raw)
                )

                # CORAL (raw)
                subj = batch["subj_idx"].long()
                L_domain = (
                    coral_loss(d2_raw, subj) +
                    coral_loss(i2_raw, subj) +
                    coral_loss(t2_raw, subj) +
                    coral_loss(l2_raw, subj)
                )

                # Reconstruction-style aux (as you had it)
                rec_loss = (
                    criterion(voxel2, batch["Voxel_Topo"].permute(0, 2, 3, 1)) +
                    criterion(region2, batch["BA_Topo"]) +
                    criterion(raw2, batch["RawBeta"])
                )

                # Mutual information aux
                v_feat = voxel2.mean((-2, -1))
                r_feat = region2.mean(-1)
                a_feat = raw2.mean(-1)
                mi_loss = (
                    compute_mutual_information(v_feat, a_feat) +
                    compute_mutual_information(r_feat, a_feat)
                )

                L_mae = args.lambda_rec * rec_loss + args.lambda_mi * mi_loss

                # classification
                L_cls = class_loss1 + class_loss2

                # total
                total_loss = (
                    args.lambda_align * L_align +
                    args.lambda_vic * L_vic +
                    args.lambda_coral * L_domain +
                    args.lambda_cls * L_cls +
                    L_mae +
                    args.lambda_sem * L_sem +
                    args.lambda_kl * kl2
                )

            # backward + step
            scaler.scale(total_loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(mae_model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # logging
            if (it % args.print_every) == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"[ep {epoch+1:02d}/{args.epochs}] it {it:04d}/{steps_per_epoch:04d} "
                    f"| m1 {mask1:.3f} m2 {mask2:.3f} lr {lr_now:.2e} "
                    f"| tot {float(total_loss.item()):.4f} "
                    f"| align {float(L_align.item()):.4f} sem {float(L_sem.item()):.4f} "
                    f"| vic {float(L_vic.item()):.4f} coral {float(L_domain.item()):.4f} "
                    f"| cls {float(L_cls.item()):.4f} rec {float(rec_loss.item()):.4f} mi {float(mi_loss.item()):.4f} "
                    f"| kl {float(kl2.item()):.4f}"
                )

        # ---- Save checkpoints ----
        payload = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model": mae_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        }

        # always save "last"
        last_path = save_checkpoint(ckpt_dir, "last.pth", payload)

        # save per-epoch if needed
        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
            ep_path = save_checkpoint(ckpt_dir, f"epoch_{epoch+1:03d}.pth", payload)
            print(f"[ckpt] saved: {ep_path}")

        dt = time.time() - t0
        print(f"[epoch] {epoch+1:03d} done in {dt/60:.1f} min | last: {last_path}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
