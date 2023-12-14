import logging
import warnings

import lovely_tensors as lt
import torch
from omegaconf import DictConfig

from src.utils import logging_utils, rich_utils

log = logging_utils.get_logger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    - Monkey-patching tensor classes to have pretty representations
    - Setting precision of float32 matrix multiplication
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable specific loggers
    if cfg.extras.get("disable_loggers"):
        disable_loggers = cfg.extras.disable_loggers
        log.info(f"Ignoring loggers! <cfg.extras.{disable_loggers=}>")
        for disable_logger in disable_loggers:
            logging.getLogger(disable_logger).setLevel(logging.ERROR)

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # monkey-patch tensor classes to have pretty representations
    if cfg.extras.get("lovely_tensors"):
        log.info("Applying monkey-patch for lovely-tensors! <cfg.extras.lovely_tensors=True>")
        lt.monkey_patch()

    # set precision of float32 matrix multiplication
    if cfg.extras.get("matmul_precision"):
        matmul_precision = cfg.extras.matmul_precision
        log.info(f"Setting precision of matrix multiplication! <cfg.extras.{matmul_precision=}>")
        torch.set_float32_matmul_precision(matmul_precision)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)
