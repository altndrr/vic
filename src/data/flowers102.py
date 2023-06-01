from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.data._base import BaseDataModule
from src.data._utils import download_data, extract_data
from src.data.components.datasets import ImageDataset


class Flowers102(BaseDataModule):
    """LightningDataModule for Flowers102 dataset.

    Statistics:
        - Around 8,000 images.
        - 102 classes.
        - URL: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/.

    Reference:
        - Nilsback et al. Automated flower classification over a large number of classes. ICCVGIP
            2008.

    Args:
        data_dir (str): Path to the data directory. Defaults to "data/".
        artifact_dir (str): Path to the artifacts directory. Defaults to "artifacts/".
    """

    name: str = "Flowers102"
    data_url: str = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"

    classes: list[str]

    def __init__(self, *args, data_dir: str = "data/", artifact_dir: str = "artifacts/", **kwargs):
        super().__init__(*args, data_dir=data_dir, **kwargs)

    @property
    def num_classes(self):
        return len(self.classes) or 102

    def prepare_data(self):
        """Download data if needed."""
        dataset_path = Path(self.hparams.data_dir, self.name)
        if dataset_path.exists():
            return

        # download data
        target_path = Path(self.hparams.data_dir, Path(self.data_url).name)
        download_data(self.data_url, target_path, from_gdrive=False)

        # rename parent folder
        archive_path = Path(self.hparams.data_dir, "102flowers.tgz")
        extract_data(archive_path)
        image_path = Path(self.hparams.data_dir, "jpg")
        image_path.rename(dataset_path)

    def setup(self, stage: Optional[str] = None):
        """Load data.

        Set variables: `self.data_train` , `self.data_val` and `self.data_test`.
        """
        if self.data_train and self.data_val and self.data_test:
            return

        dataset_path = Path(self.hparams.data_dir, self.name)
        labels_fp = Path(self.hparams.artifact_dir, "data", "flowers102", "labels.csv")
        metadata_fp = Path(self.hparams.artifact_dir, "data", "flowers102", "metadata.csv")
        split_fp = Path(self.hparams.artifact_dir, "data", "flowers102", "split_coop.csv")

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

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = Flowers102()
