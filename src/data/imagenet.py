import zipfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from rich.progress import track

from src.data._base import BaseDataModule
from src.data.components.datasets import ImageDataset

with suppress(OSError):
    import kaggle


class ImageNet(BaseDataModule):
    """LightningDataModule for ImageNet dataset.

    Statistics:
        - 1,331,167 samples.
        - 1000 classes.

    Reference:
        - Russakovsky et al. ImageNet Large Scale Visual Recognition Challenge. IJCV 2015.

    Args:
        data_dir (str): Path to the data directory. Defaults to "data/".
        train_val_split (tuple[float, float]): Train/val split ratio. Defaults to (0.9, 0.1).
        artifact_dir (str): Path to the artifacts directory. Defaults to "artifacts/".
    """

    name: str = "ImageNet"

    classes: list[str]

    data_url: str = ""

    def __init__(
        self, *args, data_dir: str = "data/", artifact_dir: str = "artifacts/", **kwargs
    ) -> None:
        super().__init__(*args, data_dir=data_dir, **kwargs)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.classes) or 1000

    def prepare_data(self) -> None:
        """Download data if needed."""
        dataset_path = Path(self.hparams.data_dir, self.name)
        if dataset_path.exists():
            return

        try:
            kaggle.api.authenticate()
        except NameError:
            raise OSError("Could not find kaggle environmental variables.")

        # download from kaggle
        kaggle.api.competition_download_files(
            "imagenet-object-localization-challenge", path=self.hparams.data_dir, quiet=False
        )

        target = Path(self.hparams.data_dir, "imagenet-object-localization-challenge.zip")
        zip_ref = zipfile.ZipFile(target, "r")

        # extract and move train images to correct folder
        train_images = [f for f in zip_ref.namelist() if "ILSVRC/Data/CLS-LOC/train/" in f]
        for filename in track(train_images, total=len(train_images)):
            zip_ref.extract(member=filename, path=Path(self.hparams.data_dir))
        output_folder = Path(self.hparams.data_dir, "ILSVRC/Data/CLS-LOC/train/")
        target_dir = Path(self.hparams.data_dir, self.name, "train")
        target_dir.mkdir(parents=True, exist_ok=True)
        output_folder.rename(target_dir)

        # extract and read the val annotations
        zip_ref.extract(member="LOC_val_solution.csv", path=Path(self.hparams.data_dir, self.name))
        val_annotations = pd.read_csv(
            Path(self.hparams.data_dir, self.name, "LOC_val_solution.csv"),
            header=0,
            index_col=0,
        )

        # # extract and move val images to correct folder
        val_images = [f for f in zip_ref.namelist() if "ILSVRC/Data/CLS-LOC/val/" in f]
        for filename in track(val_images, total=len(val_images)):
            zip_ref.extract(member=filename, path=Path(self.hparams.data_dir))
            annotations = val_annotations.loc[[Path(filename).stem]]
            folder_name = annotations["PredictionString"].item().split(" ")[0]
            file = Path(self.hparams.data_dir, filename)
            folder_dir = Path(self.hparams.data_dir, self.name, "val", folder_name)
            folder_dir.mkdir(parents=True, exist_ok=True)
            file.rename(Path(folder_dir, file.name))
        zip_ref.close()

        # remove the residue of the extraction
        Path(self.hparams.data_dir, "ILSVRC/Data/CLS-LOC/val").rmdir()
        Path(self.hparams.data_dir, "ILSVRC/Data/CLS-LOC").rmdir()
        Path(self.hparams.data_dir, "ILSVRC/Data").rmdir()
        Path(self.hparams.data_dir, "ILSVRC").rmdir()
        Path(self.hparams.data_dir, "ImageNet", "LOC_val_solution.csv").unlink()

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        if self.data_train and self.data_val and self.data_test:
            return

        dataset_path = Path(self.hparams.data_dir, self.name)
        metadata_fp = str(Path(self.hparams.artifact_dir, "data", "imagenet", "metadata.csv"))

        metadata_df = pd.read_csv(metadata_fp)
        class_names = metadata_df["class_name"].tolist()

        train_set = ImageDataset(
            str(dataset_path / "train"), class_names=class_names, transform=self.preprocess
        )
        test_set = ImageDataset(
            str(dataset_path / "val"), class_names=class_names, transform=self.preprocess
        )

        # store the list of classes
        self.classes = class_names

        self.data_train = train_set
        self.data_val = self.data_test = test_set

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
    _ = ImageNet()
