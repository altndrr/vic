from abc import ABC, abstractmethod
from typing import Optional, Union

import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.transforms import default_preprocess


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
        train_cycle_mode (str): Train cycle mode. Default to "max_size_cycle".

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
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["_metadata_"])

        assert self.name is not None, "name must be set"
        assert sum(train_val_split) == 1.0, "train_val_split must sum to 1.0"
        assert train_val_split[1] > 0, "train_val_split must have a non-zero val split"

        self._preprocess = default_preprocess(size=self.hparams.get("image_size", 224))

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return len(self.classes) or -1

    @property
    def dataloader_kwargs(self):
        """Get default kwargs for dataloader."""
        return {
            "batch_size": self.hparams.get("batch_size", 64),
            "num_workers": self.hparams.get("num_workers", 0),
            "pin_memory": self.hparams.get("pin_memory", False),
        }

    def dataloader_name(self, *args, **kwargs) -> str:
        """Get dataloader name.

        This is a useful command for logging when the {split}_dataloader returns multiple loaders.
        It is helpful to log the dataloader name with the metric.
        """
        return "dataloader_0"

    @property
    def preprocess(self):
        return self._preprocess

    @preprocess.setter
    def preprocess(self, transform: Union[list, T.Compose]):
        if not isinstance(transform, T.Compose):
            transform = T.Compose(transform)

        self._preprocess = transform

        # Propagate changes to data sets
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
    def prepare_data(self):
        """Download data if needed."""
        raise NotImplementedError

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train, **self.dataloader_kwargs, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, **self.dataloader_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, **self.dataloader_kwargs, shuffle=False)
