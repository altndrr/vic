from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.data._base import BaseDataModule
from src.data._utils import download_data, extract_data
from src.data.components.datasets import ImageDataset


class StanfordCars(BaseDataModule):
    """LightningDataModule for StanfordCars dataset.

    Statistics:
        - 16,185 images.
        - 196 classes.
        - URL: https://ai.stanford.edu/~jkrause/cars/car_dataset.html.

    Reference:
        - Krause et al. Cars: Categorization and Detection. CVPR 2013.

    Args:
        data_dir (str): Path to the data directory. Defaults to "data/".
        artifact_dir (str): Path to the artifacts directory. Defaults to "artifacts/".
    """

    name: str = "StanfordCars"

    classes: list[str]

    data_url: dict[str, str] = {
        "train": "http://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
        "test": "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
    }

    def __init__(
        self, *args, data_dir: str = "data/", artifact_dir: str = "artifacts/", **kwargs
    ) -> None:
        super().__init__(*args, data_dir=data_dir, **kwargs)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes) or 37

    def prepare_data(self) -> None:
        """Download data if needed."""
        dataset_path = Path(self.hparams.data_dir, self.name)
        if dataset_path.exists():
            return

        # download data
        for split, url in self.data_url.items():
            target_path = Path(self.hparams.data_dir, Path(url).name)
            download_data(url, target_path, from_gdrive=False)
            extract_data(target_path)
            output_path = Path(self.hparams.data_dir, f"cars_{split}")
            dataset_path.mkdir(parents=True, exist_ok=True)
            output_path.rename(dataset_path / split)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        Set variables: `self.data_train` , `self.data_val` and `self.data_test`.
        """
        if self.data_train and self.data_val and self.data_test:
            return

        dataset_path = Path(self.hparams.data_dir, self.name)
        labels_fp = Path(self.hparams.artifact_dir, "data", "stanford_cars", "labels.csv")
        metadata_fp = Path(self.hparams.artifact_dir, "data", "stanford_cars", "metadata.csv")
        split_fp = Path(self.hparams.artifact_dir, "data", "stanford_cars", "split_coop.csv")

        labels_df = pd.read_csv(labels_fp)

        metadata_df = pd.read_csv(metadata_fp)
        class_names = metadata_df["class_name"].tolist()

        split_df = pd.read_csv(split_fp)
        data = {}
        for split in ["train", "val", "test"]:
            image_paths = split_df[split_df["split"] == split]["filename"]
            merge_df = pd.merge(image_paths, labels_df, on="filename")
            image_paths = merge_df["filename"]
            image_paths = image_paths.apply(lambda x: str(dataset_path / x)).tolist()
            labels = merge_df["class_idx"].tolist()
            data[split] = ImageDataset(
                str(dataset_path),
                images=image_paths,
                labels=labels,
                class_names=class_names,
                transform=self.preprocess,
            )

        # store the class names
        self.classes = class_names

        self.data_train = data["train"]
        self.data_val = data["val"]
        self.data_test = data["test"]

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = StanfordCars()
