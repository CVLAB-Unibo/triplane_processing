import hydra
from omegaconf import OmegaConf
import wandb
from pathlib import Path
import importlib
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from datamodules.datamodule import load_datamodule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import time

code_pos = Path(__file__)

@hydra.main(config_path="cfg/triplanes_classify", config_name="main", version_base="1.1")
def main(cfg) -> None:
    print("trainers/" + cfg.trainer.module_name)
    trainer_module = importlib.import_module("trainers." + cfg.trainer.module_name)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    
    logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        dir=cfg.wandb.dir,
        save_dir=cfg.wandb.dir,
        config=OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
    )
    
    artifact = wandb.Artifact("Trainer", type="code")
    artifact.add_file("trainers/" + cfg.trainer.module_name + ".py")
    logger.experiment.log_artifact(artifact)

    ckpt_callback = ModelCheckpoint(
        dirpath= Path(cfg.wandb.dir) / cfg.wandb.run_name / "ckpts",
        filename="{epoch}-{step}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        strategy=DDPStrategy(
            find_unused_parameters=cfg.runtime.get("find_unused_parameters", True)
        )
        if cfg.runtime.gpus > 1
        else "auto",
        precision=cfg.runtime.precision,
        benchmark=True,
        logger=logger,
        callbacks=[ckpt_callback, lr_monitor],
        max_epochs=cfg.train.num_epochs,
        check_val_every_n_epoch=cfg.val.checkpoint_period,
        log_every_n_steps=200,
    )

    dm = load_datamodule(
        cfg
    )
    
    dl_train = dm.train_dataloader()
    dl_val = dm.val_dataloader()

    cfg.network.embedding_dim = cfg.train_transform.random_crop if cfg.train_transform.random_crop>0 else cfg.network.embedding_dim    
    network = hydra.utils.instantiate(cfg.network)
    print("num_params:", sum(p.numel() for p in network.parameters()))
    loss = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.opt, network.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer, cfg.opt.lr, total_steps=len(dl_train) * cfg.train.num_epochs)
        
    model = trainer_module.TrainModel(
        network=network,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg
        # train_kwargs=train_args["params"],
        # model_kwargs=model_args,
    )
    
    trainer.fit(
        model,
        train_dataloaders=dl_train,
        val_dataloaders=dl_val,
        )

    dl_test = dm.test_dataloader()
    trainer.test(ckpt_path="best", dataloaders=dl_test)
    wandb.finish()


if __name__ == "__main__":
    main()
