"""."""

import os
import subprocess
from pathlib import Path

import hydra

#import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

#from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

#from pytorch_lightning import Trainer
#from pytorch_lightning.loggers import CometLogger
#import wandb
#from src.datamodule import *

#from src.litmodule import *

"""
Sources (adapted from / inspired by):
cneuromax_old (local branch)
https://github.com/courtois-neuromod/cneuromax/blob/main/cneuromax/runner.py#L64
https://github.com/courtois-neuromod/video_transformer/blob/main/scripts/train_vqvae_ba.py
https://github.com/courtois-neuromod/phantom_LLM/blob/dev_align/phantom_LLM/src/models/run_baseline.py
https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L248
"""


@hydra.main(
    version_base=None,
    config_name="base",
    config_path="config",
)
def train(config: DictConfig) -> None:
    """.

    Args:
        config (DictConfig): .

    Returns:
        The validation loss.
    """
    print(OmegaConf.to_yaml(config))

    # Copy timeseries and extracted features .h5 files locally onto slurm (compute node local scratch)
    #subprocess.run(
    #    f"rsync -tv --info=progress2 {config.datamodule.config.features_path} $SLURM_TMPDIR/"
    #)
    #subprocess.run(
    #    f"rsync -tv --info=progress2 {config.datamodule.config.timeseries_path} $SLURM_TMPDIR/"
    #)
    #features_path = f"{os.environ["SLURM_TMPDIR"]}/{os.path.basename(config.datamodule.config.features_path)}"
    #timeseries_path = f"{os.environ["SLURM_TMPDIR"]}/{os.path.basename(config.datamodule.config.timeseries_path)}"
    #print(features_path, timeseries_path)

    #pl.seed_everything(config.random_state)

    #logger: WandbLogger = instantiate(
    #    config.logger,
    #    offline=(
    #        HydraConfig.get().launcher._target_
    #        == "hydra_plugins.hydra_submitit_launcher.submitit_launcher."
    #        "SlurmLauncher"
    #    ),
    #)
    # try instead: https://www.comet.com/site/
    #https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CometLogger.html
    # TODO: figure out comet API KEY
    #logger = CometLogger(
    #    api_key=os.environ.get("COMET_API_KEY"),
    #    workspace=os.environ.get("COMET_WORKSPACE"),  # Optional
    #    save_dir=".",  # Optional
    #    project_name="default_project",  # Optional
    #    rest_api_key=os.environ.get("COMET_REST_API_KEY"),  # Optional
    #    experiment_key=os.environ.get("COMET_EXPERIMENT_KEY"),  # Optional
    #    experiment_name="lightning_logs",  # Optional
    #)

    # instantiates datamodule config within datamodule class from config params
    #datamodule = instantiate(
    #    config.datamodule,
    #)

    #litmodule = instantiate(
    #    config.model,
    #)

    #trainer = Trainer(
    #    logger=[logger],
        # ...configs
    #)

    #trainer.fit(model=litmodule, datamodule=datamodule)

    #trainer.save_checkpoint(config.output_dir)

    #return trainer.validate(
    #    model=litmodule,
    #    datamodule=datamodule,
    #)[0]["val/loss"]


if __name__ == "__main__":
    # Retrieve the W&B key.
    #with Path("./wandb/WANDB_KEY.txt").resolve().open(
    #    "r",
    #) as f:
    #    key = f.read().strip()

    # TODO: retrieve Comet API key...

    # Login to W&B.
    #wandb.login(key=key)

    # Train (fine-tune).
    out = train()

    # If the main function returns a configuation, save it.
    #if out:
    #    OmegaConf.save(out, "out.yaml")
