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
import torchmetrics

class TrainModel(pl.LightningModule):
    def __init__(
        self,
        network = None,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler = None,
        loss = None,
        cfg = None,
        dm = None
    ):
        super().__init__()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.dataset.num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.dataset.num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.dataset.num_classes)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.net = network
        self.best_valid_acc = 0
        self.validation_step_outputs = {}
        self.validation_step_outputs["loss"] = []
        self.test_step_outputs = {}
        self.test_step_outputs["loss"] = []
        self.train_step_outputs = {}
        self.train_step_outputs["loss"] = []
        self.save_hyperparameters(cfg)
        
    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

    def forward(self, x):
        output = self.net(x)
        return output

    def training_step(self, batch, batch_idx):
        triplanes, labels = batch

        predictions = self(triplanes)
        loss = self.loss(predictions, labels)
        
        self.train_acc(F.softmax(predictions, dim=1), labels)
        self.train_step_outputs["loss"].append(loss.item())
        
        return loss
    
    def on_train_epoch_end(self):

        valid_acc = self.train_acc.compute().item()
        avg_loss_valid = np.mean(self.train_step_outputs["loss"])

        self.log_dict(
            {
                "train/loss": avg_loss_valid,
                "train/acc": valid_acc,
            }        
        )
        
        self.train_acc.reset()
        self.train_step_outputs.clear()
        self.train_step_outputs["loss"] = []    

    def validation_step(self, batch, batch_idx, dataloader_idx):
        triplanes, labels = batch
        
        predictions = self(triplanes)
        loss = self.loss(predictions, labels)

        if dataloader_idx == 0:
            self.val_acc(F.softmax(predictions, dim=1), labels)
            self.validation_step_outputs["loss"].append(loss.item())
            
        if dataloader_idx == 1:
            self.test_acc(F.softmax(predictions, dim=1), labels)
            self.test_step_outputs["loss"].append(loss.item())

        predictions = (predictions.argmax(dim=1), labels)
        return predictions
    
        
    def on_validation_epoch_end(self):

        valid_acc = self.val_acc.compute().item()
        avg_loss_valid = np.mean(self.validation_step_outputs["loss"])

        test_acc = self.test_acc.compute().item()
        avg_loss_test = np.mean(self.test_step_outputs["loss"])

        if valid_acc > self.best_valid_acc and self.global_step != 0:
            self.logger.log_metrics({"best_valid_accuracy": valid_acc})
            self.best_valid_acc = valid_acc

        self.log_dict(
            {
                "val/loss": avg_loss_valid,
                "val/acc": valid_acc,
            }        
        )
        
        self.log_dict(
            {
                "test/loss": avg_loss_test,
                "test/acc": test_acc,
            }        
        )
        
        self.val_acc.reset()
        self.test_acc.reset()
        self.validation_step_outputs.clear()
        self.test_step_outputs.clear()
        self.validation_step_outputs["loss"] = []
        self.test_step_outputs["loss"] = []
        
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, 1)
        
    def on_test_epoch_end(self):

        test_acc = self.test_acc.compute().item()
        avg_loss_test = np.mean(self.test_step_outputs["loss"])
        
        self.log_dict(
            {
                "test/final_loss": avg_loss_test,
                "test/final_acc": test_acc,
            }        
        )

    
   