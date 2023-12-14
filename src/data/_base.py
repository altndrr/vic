from abc import ABC, abstractmethod
from typing import Optional, Union

import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data._utils import default_collate_fn
from src.data.components.transforms import default_image_preprocess


class BaseDataModule(ABC, LightningDataModule):
    """LightningDataModule with base functionalities.

    Args:
        data_dir (str): Path to data directory.
        train_val_split (tuple[float, float]): Train/val split.

    Extra hparams:
        batch_size (int): Batch size. Defaults to 64.
        num_workers (int): Number of workers. Defaults to 0.
        pin_memory (bool): Pin memory. Defaults to False.
        image_size (int): Image size. Default to 224.

    Attributes:
        name (str): Name of the dataset.
        classes (list[str]): List of class names.
        data_train (Dataset): Training dataset.
        data_val (Dataset): Validation dataset.
        data_test (Dataset): Test dataset.
        num_classes (int): Number of classes.
    """

    name: str

    classes: list[str]

    def __init__(
        self,
        *args,
        data_dir: str = "data/",
        train_val_split: tuple[float, float] = (0.9, 0.1),
        **kwargs,
    ) -> None:
        super().__init__()

        assert self.name is not None, "name must be set"
        assert sum(train_val_split) == 1.0, "train_val_split must sum to 1.0"
        assert train_val_split[1] > 0, "train_val_split must have a non-zero val split"

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # save hyperparameters
        kwargs["batch_size"] = kwargs.get("batch_size", 64)
        kwargs["num_workers"] = kwargs.get("num_workers", 0)
        kwargs["pin_memory"] = kwargs.get("pin_memory", False)
        kwargs["image_size"] = kwargs.get("image_size", 224)
        self.save_hyperparameters(logger=False, ignore=["_metadata_"])

        # define default preprocess
        self._preprocess = default_image_preprocess(size=self.hparams.image_size)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes) or -1

    @property
    def dataloader_kwargs(self) -> dict:
        """Get default kwargs for dataloader."""
        return {
            "batch_size": self.hparams.batch_size,
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
            "collate_fn": default_collate_fn,
        }

    def dataloader_name(self, *args, **kwargs) -> str:
        """Get dataloader name.

        This is a useful command for logging when the {split}_dataloader returns multiple loaders.
        It is helpful to log the dataloader name with the metric.
        """
        return "dataloader_0"

    @property
    def preprocess(self) -> T.Compose:
        """Get preprocess transform."""
        return self._preprocess

    @preprocess.setter
    def preprocess(self, transform: Union[list, T.Compose]) -> None:
        """Set preprocess transform.

        Args:
            transform (Union[list, T.Compose]): Transform to be applied.
        """
        if not isinstance(transform, T.Compose):
            transform = T.Compose(transform)

        self._preprocess = transform

        # propagate changes to data sets
        for split in ["train", "val", "test"]:
            data = getattr(self, f"data_{split}")
            if data is None:
                continue

            if hasattr(data, "datasets"):
                for d in data.datasets:
                    if hasattr(d, "dataset"):
                        d.dataset.transform = self.preprocess
                    else:
                        d.transform = self.preprocess
            else:
                data.transform = self.preprocess

    @abstractmethod
    def prepare_data(self) -> None:
        """Download data if needed."""
        raise NotImplementedError

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader."""
        return DataLoader(
            dataset=self.data_train, **self.dataloader_kwargs, shuffle=True, drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(dataset=self.data_val, **self.dataloader_kwargs, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(dataset=self.data_test, **self.dataloader_kwargs, shuffle=False)
