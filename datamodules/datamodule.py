"""
Load many available datasets
"""

import random
from dataclasses import dataclass
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable
from copy import deepcopy
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
import hydra
from torchvision import transforms


def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    # base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    set_random_seed(worker_id)


def load_datamodule(
    cfg,
):

    identity_transform = transforms.Lambda(lambda x: x)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(cfg.train_transform.random_crop, antialias=True) if cfg.train_transform.random_crop > 0 else identity_transform,
            transforms.RandomHorizontalFlip() if cfg.train_transform.horizontal_flip else identity_transform,
            transforms.RandomVerticalFlip() if cfg.train_transform.vertical_flip else identity_transform,
            transforms.GaussianBlur(cfg.train_transform.gaussian_blur) if cfg.train_transform.gaussian_blur > 0 else identity_transform
            # transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(cfg.val_transoform.center_crop) if cfg.val_transoform.center_crop > 0 else identity_transform,
            # transforms.ToTensor(),
        ]),
    }

    train_ds = hydra.utils.instantiate(cfg.dataset, split="train", transform=data_transforms['train'])
    val_ds = hydra.utils.instantiate(cfg.dataset, split="val", transform=data_transforms['val'])
    
    
    # cfg2 = deepcopy(cfg.dataset)
    # cfg2.inrs_root = 'datasets/Manifold40/pcd_triplane_32_16_h64l3_freq_encoding'
    # train_ds_pc = hydra.utils.instantiate(cfg2, split="train", transform=data_transforms['train'])
    # cfg3 = deepcopy(cfg.dataset)
    # cfg3.inrs_root = 'datasets/Manifold40/voxel_triplane_32_16_h64l3_freq_encoding'
    # train_ds_voxels = hydra.utils.instantiate(cfg3, split="train", transform=data_transforms['train'])
    # train_ds = torch.utils.data.ConcatDataset([train_ds, train_ds_pc, train_ds_voxels])

    # val_ds_pc = hydra.utils.instantiate(cfg2, split="val", transform=data_transforms['val']) 
    # val_ds_voxels = hydra.utils.instantiate(cfg3, split="val", transform=data_transforms['val'])    
    # val_ds = torch.utils.data.ConcatDataset([val_ds, val_ds_pc, val_ds_voxels])

    
    test_ds = hydra.utils.instantiate(cfg.dataset, split="test",  transform=data_transforms['val'])
    
    return _DataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_batch_size=cfg.train.batch_size,
        val_batch_size=cfg.val.batch_size,
        num_workers=cfg.train.num_workers,
    )


@dataclass
class _DataModule(pl.LightningDataModule):
    train_ds: Dataset 
    val_ds: Dataset 
    test_ds: Dataset 
    train_batch_size: int = 128
    val_batch_size: int = 128
    num_workers: int = cpu_count() // 2

    def train_dataloader(self):

        train_dl = DataLoader(
            self.train_ds,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            batch_size=self.train_batch_size,
            drop_last=True,
            shuffle=True,
        )
       
        return train_dl

    def val_dataloader(self):
        
        target_dl = DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            drop_last=False,
            num_workers=self.num_workers // 2,
            worker_init_fn=worker_init_fn,
        )

        target_dl_test = DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            drop_last=False,
            num_workers=self.num_workers // 2,
        )

        return [target_dl, target_dl_test]

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            drop_last=False,
            num_workers=self.num_workers // 2,
        )
