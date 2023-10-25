from pathlib import Path
from typing import OrderedDict, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils.var import get_mlp_params_as_matrix

T_ITEM = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


class MeshInrDataset(Dataset):
    def __init__(self, inrs_root: Path, split: str, **args) -> None:
        super().__init__()

        self.inrs_root = Path(inrs_root) / split
        self.mlps_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))
        # self.sample_sd = sample_sd

    def __len__(self) -> int:
        return len(self.mlps_paths)

    def __getitem__(self, index: int) -> T_ITEM:
        with h5py.File(self.mlps_paths[index], "r") as f:
            vertices = torch.from_numpy(np.array(f.get("vertices")))
            num_vertices = torch.from_numpy(np.array(f.get("num_vertices")))
            triangles = torch.from_numpy(np.array(f.get("triangles")))
            num_triangles = torch.from_numpy(np.array(f.get("num_triangles")))
            # params = torch.from_numpy(np.array(f.get("params"))).float()
            # matrix = get_mlp_params_as_matrix(params, self.sample_sd)
            class_id = torch.from_numpy(np.array(f.get("class_id"))).long()
            # coords = torch.from_numpy(np.array(f.get("coords")))
            # labels = torch.from_numpy(np.array(f.get("labels")))

        return vertices, num_vertices, triangles, num_triangles, class_id, class_id
