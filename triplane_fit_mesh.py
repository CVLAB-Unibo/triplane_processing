import hydra
from omegaconf import OmegaConf
import wandb
from pathlib import Path
import importlib

code_pos = Path(__file__)

@hydra.main(config_path="cfg/triplane_mesh_fit", config_name="main", version_base="1.1")
def main(cfg) -> None:
    print("trainers/" + cfg.trainer.module_name)
    trainer_module = importlib.import_module("trainers." + cfg.trainer.module_name)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        dir=cfg.wandb.dir,
        config=OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
    )
    artifact = wandb.Artifact("Trainer", type="code")
    artifact.add_file("trainers/" + cfg.trainer.module_name + ".py")
    run.log_artifact(artifact)

    trainer = trainer_module.Fitter(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
