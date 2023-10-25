from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from pycarus.geometry.pcd import compute_udf_from_pcd, farthest_point_sampling
from pycarus.geometry.pcd import random_point_sampling, shuffle_pcd
from pycarus.learning.models.siren import SIREN
from pycarus.utils import progress_bar
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import open3d as o3d 
import hydra
from utils.var import get_mlps_batched_params, mlp_batched_forward, unflatten_mlp_params
from pycarus.geometry.mesh import compute_sdf_from_mesh, get_o3d_mesh_from_tensors, marching_cubes, get_tensor_mesh_from_o3d
from pycarus.metrics.chamfer_distance import chamfer_t
from pycarus.metrics.f_score import f_score
from tqdm import tqdm 
import h5py
import numpy as np
from pycarus.geometry.pcd import get_tensor_pcd_from_o3d
import wandb

class Fitter:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.out_root = Path(cfg.out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)

    def build_mlp(self, inr_params) -> SIREN:
        mlp = SIREN(
            input_dim=3,
            hidden_dim=inr_params.mlp_hidden_dim,
            num_hidden_layers=inr_params.mlp_num_hidden_layers,
            out_dim=1,
        )
        # print(sum(p.numel() for p in mlp.parameters()))
        return mlp
    
    
    def get_dataset(self, split) -> Dataset:
        dset = hydra.utils.instantiate(self.cfg.dataset, split=split)
        return dset


    def train(self) -> None: 

        
        for split in self.cfg.splits:

            dset = self.get_dataset(split)
            global_idx = 0

            loader = DataLoader(
                dset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=8,
                
            )
            desc = f"Fitting {split} set"
            print(desc)
            # c = 0
            for batch in progress_bar(loader, desc, 80):
                # if c%2 == 0:
                    # continue
                
                vertices, num_vertices, triangles, num_triangles, _, class_ids = batch
                bs = len(vertices)
                
                coords = []
                labels = []
                
                for idx in range(bs):
                    num_v = num_vertices[idx]
                    v = vertices[idx][:num_v]
                    num_t = num_triangles[idx]
                    t = triangles[idx][:num_t]
                    mesh_o3d = get_o3d_mesh_from_tensors(v, t)

                    mesh_coords, mesh_labels = compute_sdf_from_mesh(
                        mesh_o3d,
                        num_surface_points=self.cfg.num_queries_on_surface,
                        queries_stds=self.cfg.stds,
                        num_queries_per_std=self.cfg.num_points_per_std,
                        coords_range=(-1, 1),
                    )
                    coords.append(mesh_coords)
                    labels.append(mesh_labels)

                coords = torch.stack(coords, dim=0)
                labels = torch.stack(labels, dim=0)
                
                coords_and_labels = torch.cat((coords, labels.unsqueeze(-1)), dim=-1).cuda()
                coords_and_labels = shuffle_pcd(coords_and_labels)
        
                inr_params = self.cfg.inr_params
                mlps = [self.build_mlp(inr_params).cuda() for _ in range(bs)]
                batched_params = get_mlps_batched_params(mlps)
                optimizer = hydra.utils.instantiate(self.cfg.opt, batched_params)
                
                for _ in progress_bar(range(self.cfg.num_steps)):
                    selected_c_and_l = random_point_sampling(
                        coords_and_labels,
                        self.cfg.num_points_fitting,
                    )
                    
                    selected_coords = selected_c_and_l[:, :, :3]
                    selected_labels = selected_c_and_l[:, :, 3]                                    
                    pred = mlp_batched_forward(batched_params, selected_coords)

                    selected_labels = (selected_labels + 0.1) / 0.2
                    loss = F.binary_cross_entropy_with_logits(pred, selected_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                for idx in range(bs):
                    
                    flattened_params = [p[idx].view(-1) for p in batched_params]
                    flattened_params = torch.cat(flattened_params, dim=0)

                    h5_path = self.out_root / split / f"{global_idx}.h5"
                    h5_path.parent.mkdir(parents=True, exist_ok=True)

                    with h5py.File(h5_path, "w") as f:
                        f.create_dataset("vertices", data=vertices[idx].cpu().numpy())
                        f.create_dataset("num_vertices", data=num_vertices[idx])
                        f.create_dataset("triangles", data=triangles[idx].cpu().numpy())
                        f.create_dataset("num_triangles", data=num_triangles[idx])
                        f.create_dataset("class_id", data=class_ids[idx].cpu().numpy())
                        f.create_dataset("params", data=flattened_params.detach().cpu().numpy())

                    global_idx += 1