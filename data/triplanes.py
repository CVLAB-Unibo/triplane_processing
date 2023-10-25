from pathlib import Path
from typing import OrderedDict, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from utils.var import flatten_mlp_params

class MLPDataset(Dataset):
    def __init__(self, inrs_root: str, split: str, transform=None, **args) -> None:
        super().__init__()

        self.inrs_root = Path(inrs_root) / split
        self.mlps_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.mlps_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        with h5py.File(self.mlps_paths[index], "r") as f:
            params = np.array(f.get("params"))
            params = torch.from_numpy(params)
            class_id = np.array(f.get("class_id"))
            class_id = torch.from_numpy(class_id).long()

        return params, class_id

class TriplaneDataset(Dataset):
    def __init__(self, inrs_root: str, split: str, transform=None, **args) -> None:
        super().__init__()

        self.inrs_root = Path(inrs_root) / split
        self.mlps_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.mlps_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        with h5py.File(self.mlps_paths[index], "r") as f:
            triplane = np.array(f.get("triplane"))
            triplane = torch.from_numpy(triplane)
            class_id = np.array(f.get("class_id"))
            class_id = torch.from_numpy(class_id).long()
            triplane = torch.cat((triplane[0], triplane[1], triplane[2]), dim=0)
            triplane = torch.nn.functional.normalize(triplane, dim=0)

            if self.transform is not None:
                triplane = self.transform(triplane)

        return triplane, class_id


class TriplanePartDataset(Dataset):
    def __init__(self, inrs_root: str, split: str, transform=None, **args) -> None:
        super().__init__()

        self.inrs_root = Path(inrs_root) / split
        self.mlps_paths = sorted(self.inrs_root.glob("*.h5"), key=lambda x: int(x.stem))
        self.transform = transform
        self.class_to_parts = {
            "02691156": [0, 1, 2, 3],
            "02773838": [4, 5],
            "02954340": [6, 7],
            "02958343": [8, 9, 10, 11],
            "03001627": [12, 13, 14, 15],
            "03261776": [16, 17, 18],
            "03467517": [19, 20, 21],
            "03624134": [22, 23],
            "03636649": [24, 25, 26, 27],
            "03642806": [28, 29],
            "03790512": [30, 31, 32, 33, 34, 35],
            "03797390": [36, 37],
            "03948459": [38, 39, 40],
            "04099429": [41, 42, 43],
            "04225987": [44, 45, 46],
            "04379243": [47, 48, 49],
    }
        
    def __len__(self) -> int:
        return len(self.mlps_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        with h5py.File(self.mlps_paths[index], "r") as f:
            triplane = np.array(f.get("triplane"))
            triplane = torch.from_numpy(triplane)
            pcd = np.array(f.get("pcd"))
            pcd = torch.from_numpy(pcd)
            class_id = np.array(f.get("class_id"))
            class_id = torch.from_numpy(class_id).long()
            part_label = np.array(f.get("part_label"))
            part_label = torch.from_numpy(part_label).long()
            triplane = torch.cat((triplane[0], triplane[1], triplane[2]), dim=0)
            triplane = torch.nn.functional.normalize(triplane, dim=0)

            if self.transform is not None:
                triplane = self.transform(triplane)

        return triplane, class_id, part_label, pcd

