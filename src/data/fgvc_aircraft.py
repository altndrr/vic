from pathlib import Path
from shutil import rmtree
from typing import Any, Optional

from src.data._base import BaseDataModule
from src.data._utils import download_data, extract_data
from src.data.components.datasets import ImageDataset


class FGVCAircraft(BaseDataModule):
    """LightningDataModule for FGVCAircraft dataset.

    Statistics:
        - Around 10,000 images.
        - 100 classes.
        - URL: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/.

    Reference:
        - Maji et al.  Fine-Grained Visual Classification of Aircraft. Preprint 2013.

    Args:
        data_dir (str): Path to the data directory. Defaults to "data/".
        artifact_dir (str): Path to the artifacts directory. Defaults to "artifacts/".
    """

    name: str = "FGVCAircraft"

    classes: list[str]

    data_url: str = (
        "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    )

    def __init__(
        self, *args, data_dir: str = "data/", artifact_dir: str = "artifacts/", **kwargs
    ) -> None:
        super().__init__(*args, data_dir=data_dir, **kwargs)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes) or 100

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
        image_path = Path(self.hparams.data_dir, "fgvc-aircraft-2013b", "data")
        image_path.rename(dataset_path)
        rmtree(Path(self.hparams.data_dir, "fgvc-aircraft-2013b"))

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        Set variables: `self.data_train` , `self.data_val` and `self.data_test`.
        """
        if self.data_train and self.data_val and self.data_test:
            return

        dataset_path = Path(self.hparams.data_dir, self.name)

        metadata_fp = dataset_path / "variants.txt"
        class_names = [line.strip() for line in open(metadata_fp).readlines()]
        classes_to_idx = {c: i for i, c in enumerate(class_names)}

        data = {}
        for split in ["train", "val", "test"]:
            split_fp = dataset_path / f"images_variant_{split}.txt"
            with open(split_fp) as f:
                lines = [line.strip() for line in f.readlines()]
                filenames = [line.split(" ")[0] for line in lines]
                labels = [" ".join(line.split(" ")[1:]) for line in lines]

            image_paths = [str(dataset_path / "images" / f"{x}.jpg") for x in filenames]
            labels = [classes_to_idx[c] for c in labels]
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
    _ = FGVCAircraft()
