import imp
import importlib
import inspect
from statistics import median
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from pycarus.metrics.partseg_iou import PartSegmentationIoU
from pycarus.datasets.utils import get_shape_net_category_name_from_id
from utils.var import CoordsEncoder
from torch import Tensor
import wandb

class TrainModel(pl.LightningModule):
    def __init__(
        self,
        network = None,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler = None,
        loss = None,
        cfg = None,
        seg_classes = None
    ):
        super().__init__()
        self.train_iou = PartSegmentationIoU(
            use_only_category_logits=True, category_to_parts_map=seg_classes
        )
        self.val_iou = PartSegmentationIoU(
            use_only_category_logits=True, category_to_parts_map=seg_classes
        )
        self.test_iou = PartSegmentationIoU(
            use_only_category_logits=True, category_to_parts_map=seg_classes
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.net = network
        self.best_valid_iou = 0
        self.validation_step_outputs = {}
        self.validation_step_outputs["loss"] = []
        self.test_step_outputs = {}
        self.test_step_outputs["loss"] = []
        self.train_step_outputs = {}
        self.train_step_outputs["loss"] = []
        self.coords_enc = CoordsEncoder(3)
        self.save_hyperparameters(cfg)
        
    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

    def forward(self, x, class_labels, pcd):
        output = self.net(x, class_labels, pcd)
        return output

    def get_one_hot_encoding(self, x: Tensor, num_classes: int) -> Tensor:
        one_hot = torch.eye(num_classes)[x.cpu()]
        one_hot = one_hot.to(x.device)
        return one_hot

    def training_step(self, batch, batch_idx):
        triplanes, labels, part_labels, pcds = batch
        
        rand = torch.rand(pcds.shape[0], pcds.shape[1], device="cuda")
        batch_rand_perm = rand.argsort(dim=1)

        for idx in range(pcds.shape[0]):
            pcds[idx] = pcds[idx, batch_rand_perm[idx], :]
            part_labels[idx] = part_labels[idx, batch_rand_perm[idx]]

        class_labels = self.get_one_hot_encoding(labels, self.hparams.dataset.num_classes)
        coords = self.coords_enc.embed(pcds)
        out = self(triplanes, class_labels, coords)
        
        predictions = out.contiguous().view(-1, self.hparams.dataset.num_parts)
        loss = self.loss(predictions, part_labels.view(-1)  )

        self.train_iou.update(out, part_labels)
        self.train_step_outputs["loss"].append(loss.item())
        
        return loss
    
    def on_train_epoch_end(self):

        _, train_class_avg_iou, train_iou = self.train_iou.compute()
        avg_loss_valid = np.mean(self.train_step_outputs["loss"])

        self.log_dict(
            {
                "train/loss": avg_loss_valid,
                "train/iou": train_iou,
                "train/class_iou": train_class_avg_iou,
            },sync_dist=True
        )
        
        self.train_iou.reset()
        self.train_step_outputs.clear()
        self.train_step_outputs["loss"] = []    

    def validation_step(self, batch, batch_idx, dataloader_idx):
        triplanes, labels, part_labels, pcds = batch
        
        class_labels = self.get_one_hot_encoding(labels, self.hparams.dataset.num_classes)
        coords = self.coords_enc.embed(pcds)
        out = self(triplanes, class_labels, coords)
        
        predictions = out.contiguous().view(-1, self.hparams.dataset.num_parts)   
        loss = self.loss(predictions, part_labels.view(-1) )

        if dataloader_idx == 0:
            self.val_iou.update(out, part_labels)
            self.validation_step_outputs["loss"].append(loss.item())
            
        if dataloader_idx == 1:
            self.test_iou.update(out, part_labels)
            self.test_step_outputs["loss"].append(loss.item())

        # predictions = (predictions.argmax(dim=1), part_labels)
        return loss
    
        
    def on_validation_epoch_end(self):

        _, valid_class_avg_iou, valid_iou = self.val_iou.compute()
        avg_loss_valid = np.mean(self.validation_step_outputs["loss"])

        _, test_class_avg_iou, test_iou = self.test_iou.compute()
        avg_loss_test = np.mean(self.test_step_outputs["loss"])

        if valid_iou > self.best_valid_iou and self.global_step != 0:
            self.logger.log_metrics({"best_valid_iou": valid_iou})
            self.best_valid_iou = valid_iou

        self.log_dict(
            {
                "val/loss": avg_loss_valid,
                "val/iou": valid_iou,
                "val/class_iou": valid_class_avg_iou,
            } ,sync_dist=True
        )
        
        self.log_dict(
            {
                "test/loss": avg_loss_test,
                "test/iou": test_iou,
                "test/class_iou": test_class_avg_iou,
            },sync_dist=True
        )
        
        self.val_iou.reset()
        self.test_iou.reset()
        self.validation_step_outputs.clear()
        self.test_step_outputs.clear()
        self.validation_step_outputs["loss"] = []
        self.test_step_outputs["loss"] = []
        
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, 1)
        
    def on_test_epoch_end(self):

        test_mIoU_per_cat, test_class_avg_iou, test_iou = self.test_iou.compute()
        avg_loss_test = np.mean(self.test_step_outputs["loss"])
        
        self.log_dict(
            {
                "test/final_loss": avg_loss_test,
                "test/final_iou": test_iou,
                "test/final_class_iou": test_class_avg_iou,
            },sync_dist=True
        )
    
        if self.global_rank == 0:
            table = wandb.Table(columns=["class", "ioU"])
            for cat, miou in test_mIoU_per_cat.items():
                table.add_data(get_shape_net_category_name_from_id(cat), miou)
            wandb.log({"class IoU": table})

    
   