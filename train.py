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
    ]

    logger = instantiate(
        config.exp_logger,
    )
    logger.log_hyperparams(dict(config))

    trainer = instantiate(
        config.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    datamodule = instantiate(
        config.datamodule,
    )

    litmodule = instantiate(
        config.litmodule,
    )

    trainer.fit(model=litmodule, datamodule=datamodule)

    trainer.save_checkpoint(config.output_dir)

    # TODO: adapt checkpoints to saving only LoRA adapters when using LoRA
    # TODO: export accuracy metrics (val set) to plot on the brain

if __name__ == "__main__":

    """
    Implementing spawn method instead of default fork method
    https://github.com/pytorch/pytorch/issues/40403
    """
    #mp.set_start_method('spawn', force=True)

    train()
