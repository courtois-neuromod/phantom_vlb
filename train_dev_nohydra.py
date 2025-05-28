import argparse
import logging
import multiprocessing as mp
import os

#import subprocess
#from pathlib import Path
import comet_ml
from functools import partial
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.strategies import FSDPStrategy

#from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn import Conv3d, Linear
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from src import HRFConvolveLayer, RidgeRegressionLayer
from src.datamodule import (
    VLBDataModule,
    VLBDataModuleConfig,
)
from src.litmodule import (
    VLBLitModule,
    VLBLitModuleConfig,
)


def get_arguments():
    """."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--api_key', required=True, type=str)
    parser.add_argument('--workspace', required=True, type=str)
    parser.add_argument('--random_state', default=1234, type=int)
    parser.add_argument('--output_dir', default='./results/videollama2/brain_finetune/friends/lightning_ckpt', type=str)
    parser.add_argument('--cache_dir', default='./models', type=str)

    return parser.parse_args()


def train(args):
    """.

    Args:
        config (DictConfig): .

    Returns:
        The validation loss.
    """
    # set huggingface home env variable so that model weights are cached and fetched from repo
    #os.environ['HF_HOME'] = '/home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/models/'
    #os.environ["TRANSFORMERS_CACHE"] = "/home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/models"

    """ do this in bash script instead"""
    #os.environ["TRANSFORMERS_OFFLINE"] = "1"
    #os.environ['HF_HOME'] = config.cache_dir
    #os.environ["TRANSFORMERS_CACHE"] = config.cache_dir
    print(args)
    L.seed_everything(args.random_state)

    callbacks = [
        ModelCheckpoint(
            monitor="val/brain_loss",
            filename="last_brainenc",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    print(callbacks)

    logger = CometLogger(
        api_key = args.api_key,
        workspace = args.workspace,
        project_name = "phantom_mm",
        name = "vllama2_vlb_friends_logs",
        save_dir = args.output_dir,
    )
    print(logging)

    my_auto_wrap_strategy = partial(size_based_auto_wrap_policy, min_num_params=1e4)
    
    trainer = Trainer(
        precision = "16-mixed",
        accelerator = "gpu",
        gradient_clip_val = 1,
        devices = 4,
        num_nodes= 1,
        max_epochs = 1,
        max_steps = 10,
        val_check_interval = 0.5,
        log_every_n_steps = 1,
        strategy = FSDPStrategy(
            #wrapping_policy=["Linear", "Conv2d"]
            #auto_wrap_policy=auto_wrap_policy,
            #auto_wrap_policy={MistralDecoderLayer, Conv3d, Linear, HRFConvolveLayer, RidgeRegressionLayer},
            auto_wrap_policy=my_auto_wrap_strategy,
            #auto_wrap_policy=size_based_auto_wrap_policy,
            activation_checkpointing_policy={
                MistralDecoderLayer, Conv3d, Linear, HRFConvolveLayer, RidgeRegressionLayer,
            },
            state_dict_type="sharded",
            limit_all_gathers=True,
            cpu_offload=True,
        ),
        logger=logger,
        callbacks=callbacks,
    )
    
    print(trainer)
    #VLBDataModule,
    #VLBDataModuleConfig,
    # instantiates datamodule config within datamodule class from config params
    #datamodule = instantiate(
    #    config.datamodule,
    #)

    datamodule = VLBDataModule(
        VLBDataModuleConfig(
            features_path = "/home/mstlaure/projects/rrg-pbellec/mstlaure/phantom_vlb/results/videollama2/lazyloading/friends/friends_*_features.h5",
            timeseries_path = "/home/mstlaure/projects/rrg-pbellec/mstlaure/cneuromod_extract_tseries/outputs/friends/sub-03/func/sub-03_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_desc-1000Parcels7Networks_timeseries.h5",
            lazyload_path = f"{os.environ['SLURM_TMPDIR']}/friends_sub-03_*_llFile_db.h5",
            delay = 3,
            window = 3,
            seasons = ["s01", "s02", "s04", "s05", "s06"],
            subject = "sub-03",
            random_state = args.random_state,
            batch_size = 1,
            num_workers = 0,
            shuffle_val_data = True,
        ),
    )
    print(datamodule)

    litmodule = VLBLitModule(
        VLBLitModuleConfig(
            model_path = "DAMO-NLP-SG/VideoLLaMA2-7B",
            freeze_backbone = True,
                dropout_rate = 0.2,
                num_target = 1000,
                l2_lambda = 0.001,
                lr = 1e-3,
                betas = [0.9, 0.999],
                eps = 1e-08,
                weight_decay = 1e-2,
                lr_scheduler_name = "CosineAnnealingLR",
                last_epoch = -1,
                t_max = 50000,
        )
    )
    print(litmodule)

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
    #mp.set_start_method('spawn', force=True)

    args = get_arguments()
    out = train(args)

    print(out)
