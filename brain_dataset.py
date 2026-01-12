import os
import random
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.sanitize import check_and_replace_inf
from utils.hdf5_utils import first_existing_path, lazy_h5_open


class BrainDataset(Dataset):
    """
    Natural Scenes Dataset (NSD) loader with multimodal supervision.
    """

    def __init__(
        self,
        subject_ids,
        brain_file_paths,
        excel_file_paths,
        select_ba,
        is_train=False,
        is_valid=False,
        is_test=False,
        valid_ratio=0.05,
        root="E:/step02_readData",
    ):
        super().__init__()
        assert sum([is_train, is_valid, is_test]) == 1

        self.subject_ids = subject_ids
        self.root = root.replace("\\", "/")
        self.select_ba = select_ba
        self.is_train, self.is_valid, self.is_test = is_train, is_valid, is_test

        self.subj_map = {"subj01": 0, "subj02": 1, "subj05": 2, "subj07": 3}

        self.subject_indices = {}
        for subj_idx, csv_path in enumerate(excel_file_paths):
            df = pd.read_csv(csv_path)
            labels = df["shared1000"].astype(bool).values

            all_idx = list(range(min(27750, len(labels))))
            train_idx = [i for i in all_idx if not labels[i]]
            test_idx = [i for i in all_idx if labels[i]]

            random.shuffle(train_idx)
            v_sz = int(len(train_idx) * valid_ratio)
            valid_idx = train_idx[:v_sz]
            train_idx = train_idx[v_sz:]

            self.subject_indices[subj_idx] = {
                "train": train_idx,
                "valid": valid_idx,
                "test": test_idx,
            }

        split = "train" if is_train else ("valid" if is_valid else "test")
        self.indices = [
            (subj_idx, trial_idx)
            for subj_idx, d in self.subject_indices.items()
            for trial_idx in d[split]
        ]

        self._handles = {}
        self._cache = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        subj_idx, trial_idx = self.indices[idx]
        sid = self.subject_ids[subj_idx]
        rdir = f"{self.root}/{sid}"

        # ---- voxel xyz ----
        xyz_path = f"{rdir}/{sid}_interVoPos.h5"
        xyz = lazy_h5_open(self._handles, f"{sid}_xyz", xyz_path)["data"][:]
        voxel_xyz = torch.tensor(xyz, dtype=torch.float32)

        # ---- BA distance ----
        ba_path = f"{rdir}/{sid}_dist.h5"
        ba = lazy_h5_open(self._handles, f"{sid}_ba", ba_path)["cosine_distance_matrix"][:]
        ba_topo = torch.tensor(ba, dtype=torch.float32)

        # ---- Raw beta ----
        beta_path = f"{rdir}/beta-hdf5/nsdgeneral/nsdgeneral-27750x16128.hdf5"
        with h5py.File(beta_path, "r") as f:
            raw_beta = f["data"][trial_idx]
        raw_beta = raw_beta.reshape(21, 768)
        raw_beta = check_and_replace_inf(torch.tensor(raw_beta, dtype=torch.float32), "beta")

        # ---- CLIP image ----
        clip_path = first_existing_path([
            f"{rdir}/clip-w768-img-{sid}.h5py",
            f"{rdir}/p2/clip-w768-img-{sid}.h5py"
        ])
        clip = lazy_h5_open(self._handles, f"{sid}_clip", clip_path)["data"][trial_idx, 0]
        clip = torch.tensor(clip, dtype=torch.float32)

        # ---- DINO ----
        dino_path = first_existing_path([
            f"{rdir}/whole-dino-img-{sid}.h5py",
            f"{rdir}/p2/whole-dino-img-{sid}.h5py"
        ])
        dino = lazy_h5_open(self._handles, f"{sid}_dino", dino_path)["data"][trial_idx, 0]
        dino = torch.tensor(dino, dtype=torch.float32)

        return {
            "RawBeta": raw_beta,
            "Voxel_Topo": voxel_xyz,
            "BA_Topo": ba_topo,
            "CLIP_embedding": clip,
            "dino_Img": dino,
            "subj_idx": torch.tensor(self.subj_map[sid]),
        }
