from typing import Callable, Optional

import torch
import torchvision.transforms as T
from torchmetrics.aggregation import MeanMetric

from src import utils
from src.data.components.transforms import TextCompose, default_vocab_transform
from src.models.components.metrics import (
    SemanticClusterAccuracy,
    SentenceIOU,
    SentenceScore,
    UniqueValues,
)
from src.models.vocabulary_free_clip import VocabularyFreeCLIP

log = utils.get_logger(__name__)


class CaSED(VocabularyFreeCLIP):
    """LightningModule for Category Search from External Databases.

    Reference:
        Conti et al. Vocabulary-free Image Classification. NeurIPS 2023.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        vocab_transform (TextCompose, optional): List of transforms to apply to a vocabulary.
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".

    Extra hparams:
        alpha (float): Weight for the average of the image and text predictions. Defaults to 0.5.
        vocab_prompt (str): Prompt to use for a vocabulary. Defaults to "{}".
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    def __init__(self, *args, vocab_transform: Optional[TextCompose] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vocab_transform = vocab_transform or default_vocab_transform()

        # save hyperparameters
        kwargs["alpha"] = kwargs.get("alpha", 0.5)
        self.save_hyperparameters("alpha", "vocab_transform")

    @property
    def vocab_transform(self) -> Callable:
        """Get image preprocess transform.

        The getter wraps the transform in a map_reduce function and applies it to a list of images.
        If interested in the transform itself, use `self._vocab_transform`.
        """
        vocab_transform = self._vocab_transform

        def vocabs_transforms(texts: list[str]) -> list[torch.Tensor]:
            return [vocab_transform(text) for text in texts]

        return vocabs_transforms

    @vocab_transform.setter
    def vocab_transform(self, transform: T.Compose) -> None:
        """Set image preprocess transform.

        Args:
            transform (torch.nn.Module): Transform to use.
        """
        self._vocab_transform = transform

    def batch_step(
        self, images_z: torch.Tensor, vocabularies: list[list]
    ) -> tuple[torch.Tensor, list, list]:
        """Perform a single batch step.

        Args:
            images_z (torch.Tensor): Batch of image embeddings.
            images_fp (list[str]): List of paths to image files.
        """
        unfiltered_words = sum(vocabularies, [])

        # encode unfiltered words
        unfiltered_words_z = self.encode_vocabulary(unfiltered_words).squeeze(0)
        unfiltered_words_z = unfiltered_words_z / unfiltered_words_z.norm(dim=-1, keepdim=True)

        # generate a text embedding for each image from their unfiltered words
        unfiltered_words_per_image = [len(vocab) for vocab in vocabularies]
        texts_z = torch.split(unfiltered_words_z, unfiltered_words_per_image)
        texts_z = torch.stack([word_z.mean(dim=0) for word_z in texts_z])
        texts_z = texts_z / texts_z.norm(dim=-1, keepdim=True)

        # filter the words and embed them
        vocabularies = self.vocab_transform(vocabularies)
        vocabularies = [vocab or ["object"] for vocab in vocabularies]
        words = sum(vocabularies, [])
        words_z = self.encode_vocabulary(words, use_prompts=True)
        words_z = words_z / words_z.norm(dim=-1, keepdim=True)

        # create a one-hot relation mask between images and words
        words_per_image = [len(vocab) for vocab in vocabularies]
        col_indices = torch.arange(sum(words_per_image))
        row_indices = torch.arange(len(images_z)).repeat_interleave(torch.tensor(words_per_image))
        mask = torch.zeros(len(images_z), sum(words_per_image), device=self.device)
        mask[row_indices, col_indices] = 1

        # get the image and text predictions
        images_p = self.classifier(images_z, words_z, mask=mask)
        texts_p = self.classifier(texts_z, words_z, mask=mask)

        # average the image and text predictions
        samples_p = self.hparams.alpha * images_p + (1 - self.hparams.alpha) * texts_p

        return samples_p, words, vocabularies

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Lightning test step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        images = batch["images_tensor"]
        targets = batch["targets_name"]
        images_fp = batch["images_fp"]

        # get vocabularies for each image
        images_z = self.vision_encoder(images)
        images_vocab = self.vocabulary(images_z=images_z, images_fp=images_fp)

        # get predictions for each image
        images_p, words, images_vocab = self.batch_step(images_z, images_vocab)
        preds = images_p.topk(k=1, dim=-1)
        images_words = [[words[idx] for idx in indices.tolist()] for indices in preds.indices]
        images_words_values = preds.values.tolist()
        words = [
            {word: sum([v for w, v in zip(iw, iwv) if w == word]) for word in set(iw)}
            for iw, iwv in zip(images_words, images_words_values)
        ]

        # log metrics
        num_vocabs = torch.tensor([len(image_vocab) for image_vocab in images_vocab])
        num_vocabs = num_vocabs.to(self.device)
        self.metrics["test/num_vocabs_avg"](num_vocabs)
        self.log("test/num_vocabs.avg", self.metrics["test/num_vocabs_avg"])
        self.metrics["test/vocabs_unique"](images_vocab)
        self.log("test/vocabs.unique", self.metrics["test/vocabs_unique"])
        self.metrics["test/vocabs/selected_unique"](sum([list(w.keys()) for w in words], []))
        self.log("test/vocabs/selected.unique", self.metrics["test/vocabs/selected_unique"])
        self.metrics["test/semantic_iou"](words, targets)
        self.log("test/semantic_iou", self.metrics["test/semantic_iou"])
        self.metrics["test/semantic_similarity"](words, targets)
        self.log("test/semantic_similarity", self.metrics["test/semantic_similarity"])

        self.test_outputs.append((words, targets))

    def configure_metrics(self) -> None:
        """Configure metrics."""
        self.metrics["test/num_vocabs_avg"] = MeanMetric()
        self.metrics["test/vocabs_unique"] = UniqueValues()
        self.metrics["test/vocabs/selected_unique"] = UniqueValues()
        semantic_cluster_acc = SemanticClusterAccuracy(task="multiclass", average="micro")
        self.metrics["test/semantic_cluster_acc"] = semantic_cluster_acc
        self.metrics["test/semantic_iou"] = SentenceIOU()
        self.metrics["test/semantic_similarity"] = SentenceScore()


if __name__ == "__main__":
    _ = CaSED()
