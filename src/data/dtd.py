from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.data._base import BaseDataModule
from src.data._utils import download_data, extract_data
from src.data.components.datasets import ImageDataset


class DTD(BaseDataModule):
    """LightningDataModule for DTD dataset.

    Statistics:
        - 5640 images.
        - 47 classes.
        - URL: https://www.robots.ox.ac.uk/~vgg/data/dtd/.

    Reference:
        - Cimpoi et al. Describing Textures in the Wild. CVPR 2014.

    Args:
        data_dir (str): Path to the data directory. Defaults to "data/".
        artifact_dir (str): Path to the artifacts directory. Defaults to "artifacts/".
    """

    name: str = "DTD"

    classes: list[str]

    data_url: str = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"

    def __init__(
        self, *args, data_dir: str = "data/", artifact_dir: str = "artifacts/", **kwargs
    ) -> None:
        super().__init__(*args, data_dir=data_dir, **kwargs)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes) or 47

    def prepare_data(self) -> None:
        """Download data if needed."""
        dataset_path = Path(self.hparams.data_dir, self.name)
        if dataset_path.exists():
            return

        # download data
        target_path = Path(self.hparams.data_dir, Path(self.data_url).name)
        download_data(self.data_url, target_path, from_gdrive=False)

        # rename parent folder
        extract_data(target_path)
        output_path = Path(self.hparams.data_dir, "dtd")
        output_path.rename(dataset_path)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        Set variables: `self.data_train` , `self.data_val` and `self.data_test`.
        """
        if self.data_train and self.data_val and self.data_test:
            return

        dataset_path = Path(self.hparams.data_dir, self.name, "images")
        metadata_fp = Path(self.hparams.artifact_dir, "data", "dtd", "metadata.csv")
        split_fp = Path(self.hparams.artifact_dir, "data", "dtd", "split_coop.csv")

        metadata_df = pd.read_csv(metadata_fp)
        class_names = metadata_df["class_name"].tolist()
        classes_to_idx = {str(c): i for i, c in enumerate(metadata_df["folder_name"].tolist())}

        split_df = pd.read_csv(split_fp)
        data = {}
        for split in ["train", "val", "test"]:
            image_paths = split_df[split_df["split"] == split]["filename"]
            image_paths = image_paths.apply(lambda x: str(dataset_path / x)).tolist()
            folder_names = [Path(f).parent.name for f in image_paths]
            labels = [classes_to_idx[c] for c in folder_names]
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
    _ = DTD()
