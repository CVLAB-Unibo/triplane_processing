from pathlib import Path
from typing import OrderedDict, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

class PcdInrDataset(Dataset):
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
            class_id = torch.from_numpy(np.array(f.get("class_id"))).long()

        return pcd, class_id
