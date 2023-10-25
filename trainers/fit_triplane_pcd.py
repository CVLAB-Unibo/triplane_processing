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
from utils.var import get_mlps_batched_params, mlp_batched_forward, unflatten_mlp_params, CoordsEncoder
from pycarus.geometry.pcd import get_o3d_pcd_from_tensor, sample_pcds_from_udfs
from pycarus.metrics.chamfer_distance import chamfer_t
from pycarus.metrics.f_score import f_score
import h5py
import numpy as np
import wandb
class Fitter:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.out_root = Path(cfg.out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.coords_enc = CoordsEncoder(3)

    def build_mlp(self, triplane_params) -> SIREN:
        mlp = SIREN(
            input_dim=triplane_params.hidden_dim+63,
            hidden_dim=triplane_params.mlp_hidden_dim,
            num_hidden_layers=triplane_params.mlp_num_hidden_layers,
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
            
            for batch in progress_bar(loader, desc, 80):
                pcds, class_ids = batch
                bs = pcds.shape[0]
                
                if pcds.shape[1] != self.cfg.num_points_pcd:
                    pcds = farthest_point_sampling(pcds, self.cfg.num_points_pcd)

                coords = []
                labels = []
                
                for idx in range(bs):
                    pcd_coords, pcd_labels = compute_udf_from_pcd(
                        pcds[idx].to("cuda"),
                        self.cfg.num_queries_on_surface,
                        self.cfg.stds,
                        self.cfg.num_points_per_std,
                        coords_range=(-1, 1),
                        convert_to_bce_labels=True,
                    )
                    coords.append(pcd_coords)
                    labels.append(pcd_labels)

                coords = torch.stack(coords, dim=0)
                labels = torch.stack(labels, dim=0)

                coords_and_labels = torch.cat((coords, labels.unsqueeze(-1)), dim=-1).cuda()
                coords_and_labels = shuffle_pcd(coords_and_labels)                
                
                triplane_params = self.cfg.triplane_params
                triplane = torch.nn.Parameter(torch.normal(torch.zeros(bs,3,triplane_params.resolution,triplane_params.resolution,triplane_params.hidden_dim, device="cuda"), torch.ones(bs,3,triplane_params.resolution,triplane_params.resolution,triplane_params.hidden_dim, device="cuda")*0.001))

                mlps = [self.build_mlp(triplane_params).cuda() for _ in range(bs)]
                batched_params = get_mlps_batched_params(mlps)
                optimizer = hydra.utils.instantiate(self.cfg.opt, [triplane] + batched_params)
                
                for _ in progress_bar(range(self.cfg.num_steps)):
                    selected_c_and_l = random_point_sampling(
                        coords_and_labels,
                        self.cfg.num_points_fitting,
                    )
                    
                    selected_coords = selected_c_and_l[:, :, :3]
                    selected_labels = selected_c_and_l[:, :, 3]
                                    
                    coords_xy = torch.cat((selected_coords[:, :, 0].unsqueeze(2), selected_coords[:, :, 1].unsqueeze(2)), dim=2)
                    coords_xz = torch.cat((selected_coords[:, :, 0].unsqueeze(2), selected_coords[:, :, 2].unsqueeze(2)), dim=2)
                    coords_zy = torch.cat((selected_coords[:, :, 1].unsqueeze(2), selected_coords[:, :, 2].unsqueeze(2)), dim=2)

                    grid = torch.stack([coords_xy, coords_xz, coords_zy], dim=1)
                    grid = grid.reshape(-1, self.cfg.num_points_fitting, 2).unsqueeze(1)
                    features_sample = torch.nn.functional.grid_sample(triplane.reshape(-1, triplane_params.resolution, triplane_params.resolution, triplane_params.hidden_dim).permute(0, 3, 1, 2), grid).squeeze(2).squeeze(2).permute(0, 2, 1)
                    features_sample = features_sample.reshape(bs, 3, self.cfg.num_points_fitting, triplane_params.hidden_dim)
                    features_sample = features_sample.sum(1)
                    # features_sample = torch.cat([features_sample[:, 0, ...], features_sample[:, 1, ...], features_sample[:, 0, ...]], dim=-1)

                    batched_features = torch.cat([features_sample, self.coords_enc.embed(selected_coords)], dim=-1)
                    pred = mlp_batched_forward(batched_params, batched_features)
                    loss = F.binary_cross_entropy_with_logits(pred, selected_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                for idx in range(bs):
                            
                    pcd = pcds[idx]
                    class_id = class_ids[idx]

                    flattened_params = [p[idx].view(-1) for p in batched_params]
                    flattened_params = torch.cat(flattened_params, dim=0)

                    h5_path = self.out_root / split / f"{global_idx}.h5"
                    h5_path.parent.mkdir(parents=True, exist_ok=True)

                    with h5py.File(h5_path, "w") as f:
                        f.create_dataset("pcd", data=pcd.detach().cpu().numpy())
                        f.create_dataset("params", data=flattened_params.detach().cpu().numpy())
                        f.create_dataset("triplane", data=triplane[idx].permute(0,3,1,2).detach().cpu().numpy())
                        f.create_dataset("class_id", data=class_id.detach().cpu().numpy())

                    global_idx += 1