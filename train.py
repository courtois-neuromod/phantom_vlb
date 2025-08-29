import comet_ml
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_name="base",
    config_path="config",
)
def train(config: DictConfig) -> None:
    """."""
    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from src import LogValAccuracyCallback

    L.seed_everything(config.random_state)
    
    callbacks = [
        ModelCheckpoint(
            monitor="val/brain_loss",
            filename="best_brainloss_{epoch}-{step}",
            mode="min",
            dirpath=config.output_dir,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        LogValAccuracyCallback(),
    ]

    comet_logger = instantiate(
        config.comet_logger,
    )
    comet_logger.log_hyperparams(dict(config))

    cvs_logger = instantiate(
        config.cvs_logger,
    )

    trainer = instantiate(
        config.trainer,
        logger=[comet_logger, cvs_logger],
        callbacks=callbacks,
    )

    datamodule = instantiate(
        config.datamodule,
    )
    comet_logger.log_hyperparams(datamodule.datasets.dset_names)

    litmodule = instantiate(
        config.litmodule,
    )

    trainer.fit(model=litmodule, datamodule=datamodule)

    trainer.save_checkpoint(config.output_dir)

    # TODO: adapt checkpoints to saving only LoRA adapters when using LoRA
    
if __name__ == "__main__":

    """
    Implementing spawn method instead of default fork method
    https://github.com/pytorch/pytorch/issues/40403
    """
    #mp.set_start_method('spawn', force=True)

    train()
