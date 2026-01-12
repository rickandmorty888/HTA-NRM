# ==============================================================
# Alignment-only Training with VICReg + CORAL
# Uses WIC_Loss for semantic alignment
# ==============================================================

# ------------------------
# Standard library
# ------------------------
import os
import math
import inspect
import argparse

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
# Argument parser
# ==============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Align-only training with VICReg + CORAL"
    )

    # -------- Paths --------
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root directory of dataset")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/align_only",
                        help="Directory to save checkpoints")

    # -------- Training --------
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_workers", type=int, default=0)

    # -------- Loss weights --------
    parser.add_argument("--lambda_vic", type=float, default=0.05)
    parser.add_argument("--lambda_coral", type=float, default=0.05)

    # -------- Masking --------
    parser.add_argument("--mask_min", type=float, default=0.05)
    parser.add_argument("--mask_max", type=float, default=0.20)

    return parser.parse_args()


# ==============================================================
# Main training function
# ==============================================================
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)

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
        pin_memory=False,
        persistent_workers=False
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

    # Fix temperature parameters
    with torch.no_grad():
        for val, name in [
            (0.03, "logit_scale_clip"),
            (0.03, "logit_scale_dino"),
            (0.05, "logit_scale_sem1"),
            (0.05, "logit_scale_sem2"),
        ]:
            getattr(mae_model, name).copy_(
                torch.tensor(math.log(1.0 / val), device=device)
            )
            getattr(mae_model, name).requires_grad = False

    # ======================
    # Optimizer & scheduler
    # ======================
    optimizer = torch.optim.AdamW(
        [p for p in mae_model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=1e-3
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(1, args.epochs * max(1, steps_per_epoch)),
        pct_start=0.1
    )

    criterion = nn.MSELoss()

    # ======================
    # Loss helpers
    # ======================
    def vicreg_loss(z1, z2, sim_w=1.0, std_w=1.0, cov_w=0.04, eps=1e-4):
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

    def _sloss(pred, targ, stimuli, temp, margin):
        d = WIC_Loss(pred, targ, stimuli, temp=temp, margin=margin)
        return (
            2.0 * d["contrastive_loss"]
            + d["hard_negative_loss"]
            + 0.30 * d["wasserstein_loss"]
            + d["variance_loss"]
        )

    # ======================
    # Training loop
    # ======================
    def cosine01(t):
        t = max(0.0, min(1.0, t))
        return 0.5 * (1.0 - math.cos(math.pi * t))

    global_step = 0

    for epoch in range(args.epochs):
        mae_model.train()

        for it, batch in enumerate(train_loader):
            global_step += 1
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            phase = cosine01((epoch * steps_per_epoch + it + 1) /
                             max(1, args.epochs * steps_per_epoch))
            mask1 = args.mask_min + (args.mask_max - args.mask_min) * phase
            mask2 = max(args.mask_min, mask1 * (0.9 + 0.2 * torch.rand(()).item()))

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

            L_align = (
                _sloss(d1, batch["dino_Img"], batch["stimuli"], 0.03, mae_model.margin) +
                _sloss(i1, batch["CLIP_embedding"], batch["stimuli"], 0.03, mae_model.margin) +
                _sloss(t1, batch["clip_cap"], batch["stimuli"], 0.05, mae_model.margin) +
                _sloss(l1, batch["llama_cap"], batch["stimuli"], 0.05, mae_model.margin)
            )

            L_sem = _sloss(z_sem1, batch["CLIP_embedding"], batch["stimuli"], 0.03, mae_model.margin)

            L_vic = (
                vicreg_loss(d1_raw, d2_raw) +
                vicreg_loss(i1_raw, i2_raw) +
                vicreg_loss(t1_raw, t2_raw) +
                vicreg_loss(l1_raw, l2_raw)
            )

            subj = batch["subj_idx"].long()
            L_domain = (
                coral_loss(d2_raw, subj) +
                coral_loss(i2_raw, subj) +
                coral_loss(t2_raw, subj) +
                coral_loss(l2_raw, subj)
            )

            rec_loss = (
                criterion(voxel2, batch["Voxel_Topo"].permute(0, 2, 3, 1)) +
                criterion(region2, batch["BA_Topo"]) +
                criterion(raw2, batch["RawBeta"])
            )

            v_feat = voxel2.mean((-2, -1))
            r_feat = region2.mean(-1)
            a_feat = raw2.mean(-1)
            mi_loss = (
                compute_mutual_information(v_feat, a_feat) +
                compute_mutual_information(r_feat, a_feat)
            )

            L_mae = 0.05 * rec_loss + 0.03 * mi_loss
            L_cls = class_loss1 + class_loss2

            total_loss = (
                2 * L_align +
                args.lambda_vic * L_vic +
                args.lambda_coral * L_domain +
                L_cls +
                L_mae +
                L_sem +
                0.01 * kl2
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(mae_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if it % 50 == 0:
                print(
                    f"[ep {epoch+1:02d}] it {it:04d} "
                    f"loss {total_loss.item():.3f} "
                    f"align {L_align.item():.3f}"
                )

        ckpt_path = os.path.join(
            args.ckpt_dir, f"AlignOnly_VICReg_CORAL_epoch_{epoch+1}.pth"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": mae_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            ckpt_path
        )
        print(f"[ckpt] saved → {ckpt_path}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
