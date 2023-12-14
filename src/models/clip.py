from typing import Any, Optional

import open_clip
import torch
from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric

from src import utils
from src.models._base import VisionLanguageModel
from src.models.components.metrics import (
    SemanticClusterAccuracy,
    SentenceIOU,
    SentenceScore,
    UniqueValues,
)
from src.models.components.nn import LanguageTransformer, NearestNeighboursClassifier

log = utils.get_logger(__name__)


class CLIP(VisionLanguageModel):
    """LightningModule for Contrastive Language-Image Pre-training.

    Reference:
        Radford et al. Learning Transferable Visual Models From Natural Language Supervision. 2021.

    Args:
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".
        prompt (str): Prompt to use for the text encoder.
        prompts_fp (str): Path to a file containing a list of prompts. If provided, this will
            override the `prompt` argument and use a list of prompts instead.

    Extra hparams:
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    def __init__(
        self,
        *args,
        model_name: str = "RN50",
        pretrained: str = "openai",
        prompt: str = "a photo of a {}",
        prompts_fp: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._class_names = None
        self._model_name = model_name
        self._prompt = prompt
        self._prompts_fp = prompts_fp
        self._texts_views = None
        self._texts_z_views = None
        self.prompts = None

        # load model
        assert model_name in open_clip.list_models()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device="cpu"
        )

        # create submodules
        language_encoder = LanguageTransformer(
            model.transformer,
            model.token_embedding,
            model.positional_embedding,
            model.ln_final,
            model.text_projection,
            model.attn_mask,
        )
        classifier = NearestNeighboursClassifier(tau=kwargs.get("tau", 1.0))
        classifier.logit_scale = model.logit_scale

        # init base class
        super().__init__(
            *args,
            vision_encoder=model.visual,
            language_encoder=language_encoder,
            tokenizer=open_clip.get_tokenizer(model_name),
            classifier=classifier,
            **kwargs,
        )

        # set preprocess functions
        self.image_preprocess = preprocess

        # create text inputs
        self.prompts = [prompt]
        if prompts_fp is not None:
            log.info(f"Loading prompts from {prompts_fp}")
            self.prompts = open(prompts_fp).read().splitlines()

        # save hyperparameters
        kwargs["tau"] = kwargs.get("tau", 1.0)
        self.save_hyperparameters("model_name", "pretrained", "prompt", "prompts_fp", "tau")

    @property
    def texts_views(self) -> list[list[str]]:
        """Get text inputs for the text encoder.

        The number of text inputs is equal to the number of classes and the number of views is
        equal to the number of prompts.
        """
        if self._texts_views is not None:
            return self._texts_views

        if self._class_names is None:
            self._class_names = self.trainer.datamodule.classes
        self._texts_views = self.text_preprocess(self._class_names, prompts=self.prompts)

        return self._texts_views

    @property
    def texts_z_views(self) -> torch.Tensor:
        """Get text embeddings for the text encoder.

        The number of text embeddings is equal to the number of classes and the number of views is
        equal to the number of prompts.
        """
        if self._texts_z_views is not None:
            return self._texts_z_views

        texts_z_views = self.encode_text(self.texts_views)
        texts_z_views = texts_z_views / texts_z_views.norm(dim=-1, keepdim=True)
        self._texts_z_views = texts_z_views

        return self._texts_z_views

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Lightning test step.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.
        """
        images = batch["images_tensor"]
        targets = batch["targets_name"]
        classes = self.trainer.datamodule.classes

        # get vocabularies for each image
        images_z = self.vision_encoder(images)
        images_vocab = [classes] * len(images)

        # get predictions for each image
        images_p = self.classifier(images_z, self.texts_z_views)
        preds = images_p.topk(k=1, dim=-1)
        images_words = [[classes[idx] for idx in indices.tolist()] for indices in preds.indices]
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

    def on_test_epoch_end(self) -> None:
        """Lightning hook called at the end of the test epoch."""
        words, targets = zip(*self.test_outputs)
        words = sum(words, [])
        targets = sum(targets, [])
        self.metrics["test/semantic_metrics"](words, targets)
        self.log_dict(self.metrics["test/semantic_metrics"])

        super().on_test_epoch_end()

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

    @property
    def learnable_params(self) -> list[dict[str, Any]]:
        """Defines learnable parameters of the model."""
        return [{}]


if __name__ == "__main__":
    _ = CLIP()
