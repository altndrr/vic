from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import PIL
import torch
import torchvision.transforms as T
from lightning import LightningModule

from src import utils
from src.data.components.transforms import default_text_preprocess

log = utils.get_logger(__name__)


class BaseModel(ABC, LightningModule):
    """LightningModule with base functionalities.

    It is recommended to inherit this class when you want to implement your own
    LightningModule. This class provides basic functionalities, mainly for
    logging.

    Args:
        optimizer (torch.optim.Optimizer, optional): Optimizer to use.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to use.
        warmup_scheduler (torch.optim.lr_scheduler, optional): Warmup scheduler to use.
        warmup_epochs (int, optional): Number of warmup epochs. Defaults to 0.
    """

    def __init__(
        self,
        *args,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        warmup_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        warmup_epochs: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self._batch_size = None
        self._num_workers = 16
        self._image_preprocess = None
        self.metrics = torch.nn.ModuleDict()

        if optimizer is None:
            log.warning("No optimizer defined! Training will not be performed...")
        elif scheduler is None:
            log.warning("No scheduler defined! Learning rate will not be updated...")

        if warmup_scheduler is not None and warmup_epochs == 0:
            log.warning("No warmup epochs! Warmup scheduler will not be used...")
            warmup_scheduler = None

        self.save_hyperparameters("optimizer", "scheduler", "warmup_scheduler", "warmup_epochs")

        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

    @property
    @abstractmethod
    def learnable_params(self) -> list[dict[str, Any]]:
        """Get learnable parameters for the optimizer."""
        raise NotImplementedError

    @property
    def image_preprocess(self) -> Callable:
        """Get image preprocess transform.

        The getter wraps the transform in a map_reduce function and applies it to a list of images.
        If interested in the transform itself, use `self._image_preprocess`.
        """
        image_preprocess = self._image_preprocess

        def images_preprocess(images: list[PIL.Image.Image]) -> list[torch.Tensor]:
            return [image_preprocess(image) for image in images]

        return utils.map_reduce(images_preprocess, num_workers=self._num_workers, reduce="sum")

    @image_preprocess.setter
    def image_preprocess(self, transform: T.Compose) -> None:
        """Set image preprocess transform.

        Args:
            transform (torch.nn.Module): Transform to use.
        """
        self._image_preprocess = transform

        if self._trainer is not None:
            self.trainer.datamodule.preprocess = transform

    def log(self, *args, **kwargs) -> None:
        """Log a key, value pair.

        Overwrite the default log function to log on epoch end by default.
        """
        kwargs["on_step"] = kwargs.get("on_step", False)
        kwargs["on_epoch"] = kwargs.get("on_epoch", True)
        kwargs["batch_size"] = self._batch_size

        super().log(*args, **kwargs)

    def setup(self, stage: str) -> None:
        """Setup the model.

        Args:
            stage (str): Stage of the model.
        """
        super().setup(stage)

        self._batch_size = self.trainer.datamodule.hparams.batch_size
        self._num_workers = self.trainer.datamodule.hparams.num_workers
        self.trainer.datamodule.preprocess = self._image_preprocess

        self.configure_metrics()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Lightning training step.

        Args:
            batch (Any): Batch of data.
            batch_idx (int): Index of the batch.
        """
        raise NotImplementedError

    def on_train_epoch_end(self) -> None:
        """Lightning hook called at the end of the training epoch."""
        self.training_outputs.clear()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Lightning validation step.

        Args:
            batch (Any): Batch of data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        raise NotImplementedError

    def on_validation_epoch_end(self) -> None:
        """Lightning hook called at the end of the validation epoch."""
        self.validation_outputs.clear()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Lightning test step.

        Args:
            batch (Any): Batch of data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        """Lightning hook called at the end of the test epoch."""
        self.test_outputs.clear()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Lightning predict step.

        Args:
            batch (Any): Batch of data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        raise NotImplementedError

    @abstractmethod
    def configure_metrics(self) -> None:
        """Configure metrics for logging.

        Update the `self.metrics` property of type `ModuleDict` with the metrics to log. Reference
        the metrics with `self.metrics["metric_name"]`.
        """
        raise NotImplementedError

    def configure_optimizers(self) -> Optional[dict]:
        """Configure optimizers and schedulers."""
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
        tokenizer (Any): Tokenizer to use for the language input.
        classifier (torch.nn.Module): Neural network to classify the encoded inputs.
    """

    def __init__(
        self,
        *args,
        vision_encoder: torch.nn.Module,
        language_encoder: torch.nn.Module,
        tokenizer: Any,
        classifier: torch.nn.Module,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._text_preprocess = default_text_preprocess()

        self.vision_encoder: torch.nn.Module = vision_encoder
        self.language_encoder: torch.nn.Module = language_encoder
        self.tokenizer: Any = tokenizer
        self.classifier: torch.nn.Module = classifier

    @property
    def text_preprocess(self) -> Callable:
        """Get text preprocess transform.

        The getter wraps the transform in a map_reduce function and applies it to a list of texts.
        If interested in the transform itself, use `self._text_preprocess`.
        """
        return utils.map_reduce(self._text_preprocess, num_workers=self._num_workers, reduce="sum")

    @text_preprocess.setter
    def text_preprocess(self, preprocess: Callable) -> None:
        """Set text preprocess function.

        Args:
            preprocess (callable): Preprocess function to use.
        """
        self._text_preprocess = preprocess

    def encode_text(self, texts_views: list[list[str]]) -> torch.Tensor:
        """Tokenize and encode texts with the language encoder.

        Args:
            texts_views (list[list[str]]): List of texts to encode.
        """
        tokenized_texts_views = [
            torch.cat([self.tokenizer(text) for text in text_views]) for text_views in texts_views
        ]
        tokenized_texts_views = torch.stack(tokenized_texts_views).to(self.device)

        T, C, _ = tokenized_texts_views.shape
        texts_z_views = self.language_encoder(tokenized_texts_views.view(T * C, -1))
        texts_z_views = texts_z_views.view(T, C, -1)

        return texts_z_views

    @property
    def learnable_params(self) -> list[dict[str, Any]]:
        """Defines learnable parameters of the model."""
        return [
            {"name": "vision_encoder", "params": self.vision_encoder.parameters()},
            {"name": "language_encoder", "params": self.language_encoder.parameters()},
            {"name": "classifier", "params": self.classifier.parameters()},
        ]
