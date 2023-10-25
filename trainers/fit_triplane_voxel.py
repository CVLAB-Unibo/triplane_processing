from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from pycarus.geometry.pcd import compute_udf_from_pcd, farthest_point_sampling
from pycarus.geometry.pcd import random_point_sampling, shuffle_pcd, voxelize_pcd
from pycarus.learning.models.siren import SIREN
from pycarus.utils import progress_bar
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import open3d as o3d 
import hydra
from utils.var import get_mlps_batched_params, mlp_batched_forward, unflatten_mlp_params, CoordsEncoder, focal_loss
from pycarus.geometry.pcd import get_o3d_pcd_from_tensor, sample_pcds_from_udfs
from pycarus.metrics.chamfer_distance import chamfer_t
from pycarus.metrics.f_score import f_score
import h5py
import numpy as np
import wandb
from einops import rearrange, repeat

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
                pcds = pcds.to("cuda")
                bs = pcds.shape[0]
                
                if pcds.shape[1] != self.cfg.num_points_pcd:
                    pcds = farthest_point_sampling(pcds, self.cfg.num_points_pcd)

                vgrids, centroids = voxelize_pcd(pcds, self.cfg.vox_res, -1, 1)

                coords = repeat(centroids, "r1 r2 r3 d -> b r1 r2 r3 d", b=bs)
                coords = rearrange(coords, "b r1 r2 r3 d -> b (r1 r2 r3) d")
                labels = rearrange(vgrids, "b r1 r2 r3 -> b (r1 r2 r3)")

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

                    batched_features = torch.cat([features_sample, self.coords_enc.embed(selected_coords)], dim=-1)
                    pred = mlp_batched_forward(batched_params, batched_features)
                    loss = focal_loss(pred, selected_labels)
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


    def evaluate(self) -> None:
        triplane_params = self.cfg.triplane_params
        mlp = self.build_mlp(triplane_params).cuda()
        
        for split in self.cfg.splits:
        
            print(f'evaluating {split}')
            
            cdts = []
            fscores = []
            
            dset_root =  (Path(self.cfg.out_root) / split).glob("*.h5")
            mlps_paths = sorted(list(dset_root), key=lambda p: int(p.stem))
            print(f"evaluating with {len(mlps_paths)} shapes")
            
            for idx in progress_bar(range(len(mlps_paths))):

                with h5py.File(mlps_paths[idx], "r") as f:
                    gt_pc = torch.from_numpy(np.array(f.get("pcd")))
                    params = torch.from_numpy(np.array(f.get("params")))
                    triplane = torch.from_numpy(np.array(f.get("triplane"))).cuda()       
                   
                mlp.load_state_dict(unflatten_mlp_params(params, mlp.state_dict()))         
                         
                ##visualize
                gt_vgrid, centroids = voxelize_pcd(gt_pc, self.cfg.vox_res, -1, 1)
                pcd_gt = centroids[gt_vgrid == 1]

                centr = centroids.unsqueeze(0).cuda()
                centr = rearrange(centr, "b r1 r2 r3 d -> b (r1 r2 r3) d")
                
                coords = centr.squeeze(0)
                coords_xy = torch.cat((coords[:, 0].unsqueeze(1), coords[:, 1].unsqueeze(1)), dim=1)
                coords_xz = torch.cat((coords[:, 0].unsqueeze(1), coords[:, 2].unsqueeze(1)), dim=1)
                coords_zy = torch.cat((coords[:, 1].unsqueeze(1), coords[:, 2].unsqueeze(1)), dim=1)
                grid = torch.stack([coords_xy, coords_xz, coords_zy], dim=0).unsqueeze(1).cuda()
                features = torch.nn.functional.grid_sample(triplane, grid).squeeze(2).permute(0, 2, 1)
                features = features.sum(0)
                features = torch.cat([features, self.coords_enc.embed(coords)], dim=-1)
                
                vgrid_pred = torch.sigmoid(mlp(features)[0])

                vgrid_pred = vgrid_pred.squeeze(1)
                centr = centr.squeeze(0)

                vgrid_pred_npz = torch.zeros_like(gt_vgrid)
                vgrid_pred_npz[vgrid_pred.reshape(64,64,64) > 0.3] = 1
                np.savez(f'reconstrauctions_shapenet_vox/{idx}', voxel=vgrid_pred_npz)
                                
                mlp_pcd = centr[vgrid_pred > 0.3]
                mlp_pcd_o3d = get_o3d_pcd_from_tensor(mlp_pcd.cpu())
                o3d.io.write_point_cloud(f'reconstrauctions_shapenet_vox/{idx}.pcd', mlp_pcd_o3d)

                cd = chamfer_t(mlp_pcd, pcd_gt.cuda())
                cdts.append(cd.item())
                f = f_score(mlp_pcd, pcd_gt.cuda(), threshold=0.01)[0]
                fscores.append(f.item())
        
            mean_cdt = sum(cdts) / len(cdts)
            mean_fscore = sum(fscores) / len(fscores)
            print("mean CD", mean_cdt)
            print("mean fscore", mean_fscore)
            wandb.log({f"mean_cdt_{split}": mean_cdt, f"mean_fscore_{split}": mean_fscore})