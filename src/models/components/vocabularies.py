import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

from src.models.components.retrieval import RetrievalDatabase, download_retrieval_databases

__all__ = ["BLIP2VQAVocabulary", "ImageNetVocabulary", "RetrievalVocabulary"]


class BaseVocabulary(ABC, torch.nn.Module):
    """Base class for a vocabulary for image classification."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> list[list[str]]:
        values = self.forward(*args, **kwargs)
        return values

    @abstractmethod
    def forward(self, *args, **kwargs) -> list[list[str]]:
        """Forward pass."""
        raise NotImplementedError


class BLIP2VQAVocabulary(BaseVocabulary):
    """Vocabulary based on VQA with BLIP2 on images."""

    def __init__(
        self,
        *args,
        model_name: str = "blip2_t5",
        model_type: str = "pretrain_flant5xl",
        question: str = "Question: what's in the image? Answer:",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        text_generator, image_preprocess, _ = load_model_and_preprocess(
            model_name, model_type=model_type, is_eval=True
        )
        self.model_name = model_name
        self.model_type = model_type
        self._text_generator = text_generator.float()
        self._text_generator_image_preprocess = image_preprocess["eval"]
        self._question = question

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return list(self._text_generator.parameters())[0].device

    @torch.no_grad()
    def forward(self, *args, images_fp: Optional[list[str]] = None, **kwargs) -> list[list[str]]:
        """Create a vocabulary for a batch of images.

        Args:
            images_fp (list[str]): Path to image files to create vocabularies for. Defaults to
                None.
        """
        assert images_fp is not None

        images = [Image.open(fp).convert("RGB") for fp in images_fp]
        images = torch.stack([self._text_generator_image_preprocess(image) for image in images])
        images = images.to(self.device)

        # generate captions
        captions = self._text_generator.generate({"image": images, "prompt": self._question})

        # split captions into vocabularies
        vocabularies = [list(set(caption.split(","))) for caption in captions]

        # clean vocabularies
        vocabularies = [[v.strip().lower() for v in vocabulary] for vocabulary in vocabularies]
        vocabularies = [[v for v in vocabulary if v] for vocabulary in vocabularies]

        # fill empty vocabularies with a single word
        vocabularies = [["object"] if not v else v for v in vocabularies]

        return vocabularies


class ImageNetVocabulary(BaseVocabulary):
    """Vocabulary based on ImageNet classes.

    Args:
        artifact_dir (str): Path to the artifacts directory. Defaults to "artifacts/".
    """

    def __init__(self, *args, artifact_dir: str = "artifacts/", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._artifact_dir = artifact_dir

        metadata_fp = str(Path(self._artifact_dir, "data", "imagenet", "metadata.csv"))
        metadata_df = pd.read_csv(metadata_fp)
        class_names = metadata_df["class_name"].tolist()

        self._words = class_names

    def forward(self, *args, **kwargs) -> list[list[str]]:
        """Create a vocabulary for a batch of images."""
        batch_size = max(len(kwargs.get("images_z", kwargs.get("images_fp", []))), 1)

        return [self._words] * batch_size


class RetrievalVocabulary(BaseVocabulary):
    """Vocabulary based on captions from an external database.

    Args:
        database_name (str): Name of the database to use.
        databases_dict_fp (str): Path to the databases dictionary file.
        num_samples (int): Number of samples to return. Default is 40.
    """

    def __init__(
        self, *args, database_name: str, databases_dict_fp: str, num_samples: int = 10, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.database_name = database_name
        self.databases_dict_fp = databases_dict_fp
        self.num_samples = num_samples

        with open(databases_dict_fp, encoding="utf-8") as f:
            databases_dict = json.load(f)

        download_retrieval_databases()
        self.database = RetrievalDatabase(databases_dict[database_name])

    def __call__(self, *args, **kwargs) -> list[list[str]]:
        values = super().__call__(*args, **kwargs)

        # keep only the `num_samples` first words
        num_samples = self.num_samples
        values = [value[:num_samples] for value in values]

        return values

    def forward(self, *args, images_z: Optional[torch.Tensor] = None, **kwargs) -> list[list[str]]:
        """Create a vocabulary for a batch of images.

        Args:
            images_z (torch.Tensor): Batch of image embeddings.
        """
        assert images_z is not None

        images_z = images_z / images_z.norm(dim=-1, keepdim=True)
        images_z = images_z.cpu().detach().numpy().tolist()

        if isinstance(images_z[0], float):
            images_z = [images_z]

        query = np.matrix(images_z).astype("float32")
        results = self.database.query(query, modality="text", num_samples=self.num_samples)
        vocabularies = [[r["caption"] for r in result] for result in results]

        return vocabularies
