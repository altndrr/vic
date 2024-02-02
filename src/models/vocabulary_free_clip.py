import torch
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric

from src import utils
from src.models.clip import CLIP
from src.models.components.metrics import (
    SemanticClusterAccuracy,
    SentenceIOU,
    SentenceScore,
    UniqueValues,
)
from src.models.components.vocabularies import BaseVocabulary, ImageNetVocabulary

log = utils.get_logger(__name__)


class VocabularyFreeCLIP(CLIP):
    """LightningModule for Contrastive Language-Image Pre-training without a vocabulary.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".

    Extra hparams:
        vocab_prompt (str): Prompt to use for a vocabulary. Defaults to "{}".
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    def __init__(
        self,
        *args,
        vocabulary: BaseVocabulary = ImageNetVocabulary(),
        vocab_prompt: str = "{}",
        **kwargs,
    ) -> None:
        super().__init__(*args, prompt="{}", **kwargs)
        self.vocabulary = vocabulary
        self.vocab_prompts = [vocab_prompt]
        self._prev_vocab_words = None
        self._prev_used_prompts = None
        self._prev_vocab_words_z = None

    def encode_vocabulary(self, vocab: list[str], use_prompts: bool = False) -> torch.Tensor:
        """Encode a vocabulary.

        Args:
            vocab (list): List of words.
        """
        if vocab == self._prev_vocab_words and use_prompts == self._prev_used_prompts:
            return self._prev_vocab_words_z

        prompts = self.vocab_prompts if use_prompts else None
        texts_z_views = self.encode_text(self.text_preprocess(vocab, prompts=prompts))

        # cache vocabulary
        self._prev_vocab_words = vocab
        self._prev_used_prompts = use_prompts
        self._prev_vocab_words_z = texts_z_views

        return texts_z_views

    def batch_step(
        self, images_z: torch.Tensor, vocabularies: list[list]
    ) -> tuple[torch.Tensor, list, list]:
        """Perform a single batch step.

        Args:
            images_z (torch.Tensor): Batch of image embeddings.
            images_fp (list[str]): List of paths to image files.
        """
        words = sum(vocabularies, [])

        # encode vocabularies
        words_z = self.encode_vocabulary(words).squeeze(0)
        words_z = words_z / words_z.norm(dim=-1, keepdim=True)

        # create a one-hot relation mask between images and words
        words_per_image = [len(vocab) for vocab in vocabularies]
        col_indices = torch.arange(sum(words_per_image))
        row_indices = torch.arange(len(images_z)).repeat_interleave(torch.tensor(words_per_image))
        mask = torch.zeros(len(images_z), sum(words_per_image), device=self.device)
        mask[row_indices, col_indices] = 1

        images_p = self.classifier(images_z, words_z, mask=mask)

        return images_p, words, vocabularies

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Lightning test step.

        Args:
            batch (Any): Batch of data.
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

        self.test_outputs.append((words, targets))

    def configure_metrics(self) -> None:
        """Configure metrics."""
        self.metrics["test/num_vocabs_avg"] = MeanMetric()
        self.metrics["test/vocabs_unique"] = UniqueValues()
        self.metrics["test/vocabs/selected_unique"] = UniqueValues()

        semantic_metrics = {}
        semantic_cluster_acc = SemanticClusterAccuracy(task="multiclass", average="micro")
        semantic_metrics["test/semantic_cluster_acc"] = semantic_cluster_acc
        semantic_metrics["test/semantic_iou"] = SentenceIOU()
        semantic_metrics["test/semantic_similarity"] = SentenceScore()
        self.metrics["test/semantic_metrics"] = MetricCollection(semantic_metrics)


if __name__ == "__main__":
    _ = VocabularyFreeCLIP()
