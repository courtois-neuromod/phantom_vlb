#import logging
#import multiprocessing as mp
#import os

#import subprocess
#from pathlib import Path
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
    """.

    Args:
        config (DictConfig): .

    Returns:
        The validation loss.
    """
    # Debugging sanity checks
    #logging.info(config)

    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

    L.seed_everything(config.random_state)
    
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

    #logging.info(
    #    f"Best model saved at {callbacks[0].best_model_path}, \
    #       with a val loss of {callbacks[0].best_model_score}"
    #)
    #trainer.save_checkpoint(config.output_dir)

    # TODO: implement LoRA
    # https://github.com/courtois-neuromod/phantom_LLM/blob/dev_beluga/phantom_LLM/src/utils.py
    # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/scripts/custom/finetune_lora.sh
    # --lora_enable True --lora_r 128 --lora_alpha 256
    # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/train.py
    # https://gemini.google.com/app/41844d6e03d7786e
    """
     # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"   
    """


if __name__ == "__main__":

    """
    Implementing spawn method instead of default fork method
    https://github.com/pytorch/pytorch/issues/40403
    """
    #mp.set_start_method('spawn', force=True)

    train()
