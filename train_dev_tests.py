"""
Sources (adapted from / inspired by):
cneuromax_old (local branch)
https://github.com/courtois-neuromod/cneuromax/blob/main/cneuromax/runner.py#L64
https://github.com/courtois-neuromod/video_transformer/blob/main/scripts/train_vqvae_ba.py
https://github.com/courtois-neuromod/phantom_LLM/blob/dev_align/phantom_LLM/src/models/run_baseline.py
https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/99bce703036a498f8e76a2adb9fd3f50c969beb0/videollama2/train.py#L248
"""

import logging
import multiprocessing as mp
import os

#import subprocess
#from pathlib import Path
import comet_ml
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_name="base_dev",
    config_path="config",
)
def train(config: DictConfig) -> None:
    """.

    Args:
        config (DictConfig): .

    Returns:
        The validation loss.
    """
    # Debugging sanity checks
    logging.info(config)

    print(OmegaConf.to_yaml(config))
    print(config.subject, config.random_state)
    print(config.datamodule.config.subject, config.datamodule.config.random_state)

    # set huggingface home env variable so that model weights are cached and fetched from repo
    #os.environ['HF_HOME'] = '/home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/models/'
    #os.environ["TRANSFORMERS_CACHE"] = "/home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/models"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ['HF_HOME'] = config.cache_dir
    #os.environ["TRANSFORMERS_CACHE"] = config.cache_dir

    import pytorch_lightning as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.strategies import FSDPStrategy

    #from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from torch.nn import Conv3d, Linear
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

    """
    Lightning training on multiple GPUs
    https://pytorch-lightning.readthedocs.io/en/0.8.5/multi_gpu.html

    Lightning strategies
    https://lightning.ai/docs/pytorch/stable/extensions/strategy.html
    https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html
    """

    #from lightning.pytorch.loggers.wandb import WandbLogger
    #from pytorch_lightning import Trainer
    #from pytorch_lightning.loggers import CometLogger

    # UPDATE / hack : Files now copied in bash command line before launching script
    # Copy timeseries and extracted features .h5 files locally onto slurm (compute node local scratch)
    #subprocess.run(
    #    f"rsync -tv --info=progress2 {config.datamodule.config.features_path} $SLURM_TMPDIR/"
    #)
    #subprocess.run(
    #    f"rsync -tv --info=progress2 {config.datamodule.config.timeseries_path} $SLURM_TMPDIR/"
    #)
    #config.datamodule.config.features_path = f"{os.environ["SLURM_TMPDIR"]}/{os.path.basename(config.datamodule.config.features_path)}"
    #config.datamodule.config.timeseries_path = f"{os.environ["SLURM_TMPDIR"]}/{os.path.basename(config.datamodule.config.timeseries_path)}"
    #print(features_path, timeseries_path)

    pl.seed_everything(config.random_state)

    # try comet logger: https://www.comet.com/site/
    #
    # Comet quick quide
    # https://www.comet.com/mariestlaurent/quick-start?fromGetStarted=true
    #
    # https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CometLogger.html
    # TODO: figure out comet API KEY / offline mode (save local dir)
    #logger = CometLogger(
    #    api_key=os.environ.get("COMET_API_KEY"),
    #    workspace=os.environ.get("COMET_WORKSPACE"),  # Optional
    #    save_dir=".",  # Optional
    #    project_name="default_project",  # Optional
    #    rest_api_key=os.environ.get("COMET_REST_API_KEY"),  # Optional
    #    experiment_key=os.environ.get("COMET_EXPERIMENT_KEY"),  # Optional
    #    experiment_name="lightning_logs",  # Optional
    #)

    """
    Adapted from: https://github.com/courtois-neuromod/video_transformer/blob/0906e9a71a2fdb511190f7a757c8aadcb1f6c990/scripts/train_gpt_ba.py#L33
    """
    callbacks = [
        ModelCheckpoint(
            monitor="val/brain_loss",
            filename="last_brainenc",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = instantiate(
        config.exp_logger,
        #save_dir=f"{os.getcwd()}/logs",  # only for offline mode, if I understand
    )
    logger.log_hyperparams(dict(config))

    """
    doc on params:
    https://lightning.ai/docs/pytorch/stable/common/trainer.html
    https://github.com/pytorch/pytorch/blob/7cf8049d63ee7e8632f2e7332a654d4f2b9550a4/torch/distributed/fsdp/wrap.py#L143
    https://www.restack.io/p/pytorch-lightning-answer-fsdp-tutorial-cat-ai
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html
    https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
    https://docs.pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
    https://github.com/pytorch/pytorch/blob/7cf8049d63ee7e8632f2e7332a654d4f2b9550a4/torch/distributed/fsdp/wrap.py#L295
    """
    #auto_wrap_policy = transformer_auto_wrap_policy(
    #    {MistralDecoderLayer},
    #)

    trainer = instantiate(
        config.trainer,
        strategy=FSDPStrategy(
            #wrapping_policy=["Linear", "Conv2d"]
            #auto_wrap_policy=auto_wrap_policy,
            #auto_wrap_policy={MistralDecoderLayer, Conv3d, Linear},
            auto_wrap_policy=size_based_auto_wrap_policy,
            activation_checkpointing_policy={MistralDecoderLayer, Conv3d, Linear},
            cpu_offload=True,
        ),
        logger=logger,
        callbacks=callbacks,
    )

    # instantiates datamodule config within datamodule class from config params
    datamodule = instantiate(
        config.datamodule,
    )

    litmodule = instantiate(
        config.litmodule,
    )

    trainer.fit(model=litmodule, datamodule=datamodule)

    #logging.info(
    #    f"Best model saved at {callbacks[0].best_model_path}, \
    #       with a val loss of {callbacks[0].best_model_score}"
    #)
    #trainer.save_checkpoint(config.output_dir)

    # TODO: implement LoRA
    # TODO: Comet: how to log, save, view results (offline mode)
    # https://www.comet.com/mariestlaurent/quick-start?fromGetStarted=true
    # https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CometLogger.html

    return trainer.validate(
        model=litmodule,
        datamodule=datamodule,
    )[0]["val/brain_loss"]


if __name__ == "__main__":

    """
    Implementing spawn method instead of default fork method
    https://github.com/pytorch/pytorch/issues/40403
    """
    mp.set_start_method('spawn', force=True)

    # Train (fine-tune).
    out = train()

    # If the main function returns a configuation, save it.
    if out:
        OmegaConf.save(out, "out.yaml")
