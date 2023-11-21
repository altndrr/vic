import hydra
import pyrootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

root = pyrootutils.setup_root(search_from=__file__, indicator="pyproject.toml", pythonpath=True)

from src import utils

log = utils.get_logger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict, dict]:
    """Evaluates given checkpoint on a dataset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multi-runs, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    log.info(f"Instantiating data <{cfg.data._target_}>")
    data: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.ckpt_path is None:
        log.warning("No checkpoint path provided! Using the initial weights...")

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {"cfg": cfg, "data": data, "model": model, "logger": logger, "trainer": trainer}

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=data, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entrypoint for evaluation.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
