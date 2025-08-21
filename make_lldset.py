import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_name="base",
    config_path="config",
)
def make_lldset(config: DictConfig) -> None:
    """.

    Args:
        config (DictConfig): .
    """
    import lightning as L
    L.seed_everything(config.random_state)
    
    datamodule = instantiate(
        config.datamodule,
    )


if __name__ == "__main__":

    """
    Script generates temporary train and val dsets per subject
    for lazy loading, to speed up training.
    """

    make_lldset()
