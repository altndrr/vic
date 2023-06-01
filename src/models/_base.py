from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from lightning import LightningModule

from src import utils

log = utils.get_logger(__name__)


class BaseModel(ABC, LightningModule):
    """LightningModule with base functionalities.

    It is recommended to inherit this class when you want to implement your own
    LightningModule. This class provides basic functionalities, mainly for
    logging and metrics.


    Args:
        optimizer (torch.optim.Optimizer, optional): Optimizer to use.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to use.

    Extra hparams:
        custom_preprocess (list | T.Compose)): Custom data preprocess to use.
    """

    def __init__(
        self,
        *args,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        warmup_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        warmup_epochs: int = 0,
        **kwargs,
    ):
        super().__init__()

        if optimizer is None:
            log.warning("No optimizer defined! Training will not be performed...")
        elif scheduler is None:
            log.warning("No scheduler defined! Learning rate will not be updated...")

        if warmup_scheduler is not None and warmup_epochs == 0:
            log.warning("No warmup epochs! Warmup scheduler will not be used...")
            warmup_scheduler = None

        self.save_hyperparameters("optimizer", "scheduler", "warmup_scheduler", "warmup_epochs")
        if "custom_preprocess" in kwargs:
            self.save_hyperparameters("custom_preprocess")

        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

    @property
    @abstractmethod
    def learnable_params(self) -> list[dict[str, Any]]:
        """Get learnable parameters for the optimizer."""
        raise NotImplementedError

    def log(self, *args, **kwargs):
        """Log a key, value pair.

        Overwrite the default log function to log on epoch end by default.
        """
        kwargs["on_step"] = False
        kwargs["on_epoch"] = True
        kwargs["batch_size"] = self._batch_size

        super().log(*args, **kwargs)

    def setup(self, stage: str) -> None:
        if self.hparams.get("custom_preprocess") is not None:
            log.info("Overwrite data preprocessing with custom model transforms...")
            self.trainer.datamodule.preprocess = self.hparams.custom_preprocess
        self._batch_size = self.trainer.datamodule.hparams.batch_size

    def training_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def on_train_epoch_end(self) -> None:
        self.training_outputs.clear()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        raise NotImplementedError

    def on_validation_epoch_end(self) -> None:
        self.validation_outputs.clear()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        self.test_outputs.clear()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        raise NotImplementedError

    def configure_optimizers(self):
        if self.hparams.get("optimizer") is None:
            return None

        for param in self.parameters():
            param.requires_grad = False

        for nn in self.learnable_params:
            for p in nn["params"]:
                p.requires_grad = True

        optimizer = self.hparams.optimizer(params=self.learnable_params)
        if self.hparams.get("scheduler") is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            if self.hparams.get("warmup_epochs", 0) > 0:
                warmup_scheduler = self.hparams.warmup_scheduler(optimizer=optimizer)
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[self.hparams.warmup_epochs],
                )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class VisionLanguageModel(BaseModel):
    """LightningModule for vision and language tasks.

    It is recommended to inherit this class when you want to implement your own
    LightningModule for vision and language tasks.

    Args:
        vision_encoder (torch.nn.Module): Neural network to encode the vision input.
        language_encoder (torch.nn.Module): Neural network to encode the language input.
        classifier (torch.nn.Module): Neural network to classify the encoded inputs.
    """

    def __init__(
        self,
        *args,
        vision_encoder: torch.nn.Module,
        language_encoder: torch.nn.Module,
        classifier: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.vision_encoder: torch.nn.Module = vision_encoder
        self.language_encoder: torch.nn.Module = language_encoder
        self.classifier: torch.nn.Module = classifier

    @property
    def learnable_params(self) -> list[dict[str, Any]]:
        """Defines learnable parameters of the model."""
        return [
            {"name": "vision_encoder", "params": self.vision_encoder.parameters()},
            {"name": "language_encoder", "params": self.language_encoder.parameters()},
            {"name": "classifier", "params": self.classifier.parameters()},
        ]
