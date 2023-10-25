from pathlib import Path
from typing import OrderedDict, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pycarus.geometry.pcd import voxelize_pcd

class VoxelInrDataset(Dataset):
    def __init__(self, inrs_root: str, split: str, **args) -> None:
        super().__init__()

        self.inrs_root = Path(inrs_root) / split
        self.mlps_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))
        
    def __len__(self) -> int:
        return len(self.mlps_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        with h5py.File(self.mlps_paths[index], "r") as f:
            pcd = torch.from_numpy(np.array(f.get("pcd")))
            # params = np.array(f.get("params"))
            # params = torch.from_numpy(params).float()
            # matrix = get_mlp_params_as_matrix(params, self.sample_sd)
            class_id = torch.from_numpy(np.array(f.get("class_id"))).long()

        # vgrid, centroids = voxelize_pcd(pcd, self.vox_res, -1, 1)

        return pcd, class_id
