from typing import Any, Optional

import open_clip
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.aggregation import MeanMetric

from src import utils
from src.models._base import VisionLanguageModel
from src.models.components.metrics import (
    SemanticClusterAccuracy,
    SemanticIOU,
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
    ):
        self.metrics = None
        self.preprocess = None
        self.prompts = None
        self._class_names = None
        self._model_name = model_name
        self._prompt = prompt
        self._prompts_fp = prompts_fp
        self._texts_views = None
        self._texts_z_views = None

        # load model
        assert model_name in open_clip.list_models()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device="cpu"
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        self.tokenizer = tokenizer
        self.preprocess = preprocess

        # create text inputs
        self.prompts = [prompt]
        if prompts_fp is not None:
            log.info(f"Loading prompts from {prompts_fp}")
            self.prompts = open(prompts_fp).read().splitlines()

        # save hyperparameters
        kwargs["tau"] = kwargs.get("tau", 1.0)
        self.save_hyperparameters("model_name", "pretrained", "prompt", "prompts_fp", "tau")

        # create submodules
        language_encoder = LanguageTransformer(
            model.transformer,
            model.token_embedding,
            model.positional_embedding,
            model.ln_final,
            model.text_projection,
            model.attn_mask,
        )
        scale = model.logit_scale.exp().item()
        classifier = NearestNeighboursClassifier(scale=scale, tau=self.hparams.tau)

        # init base class
        super().__init__(
            *args,
            vision_encoder=model.visual,
            language_encoder=language_encoder,
            classifier=classifier,
            custom_preprocess=preprocess,
            **kwargs,
        )

    @property
    def texts_views(self) -> torch.Tensor:
        """Get text inputs for the text encoder.

        The number of text inputs is equal to the number of classes and the number of views is
        equal to the number of prompts.
        """
        if self._texts_views is not None:
            return self._texts_views

        if self._class_names is None:
            self._class_names = self.trainer.datamodule.classes
        classes = [c.replace("_", " ") for c in self._class_names]
        texts_views = [[p.format(c) for c in classes] for p in self.prompts]
        self._texts_views = texts_views

        return self._texts_views

    @property
    def texts_z_views(self) -> torch.Tensor:
        """Get text embeddings for the text encoder.

        The number of text embeddings is equal to the number of classes and the number of views is
        equal to the number of prompts.
        """
        if self._texts_z_views is not None:
            return self._texts_z_views

        tokenized_texts_views = [
            torch.cat([self.tokenizer(prompt) for prompt in text_views]).to(self.device)
            for text_views in self.texts_views
        ]
        tokenized_texts_views = torch.stack(tokenized_texts_views)

        T, C, _ = tokenized_texts_views.shape
        texts_z_views = self.language_encoder(tokenized_texts_views.view(T * C, -1))
        texts_z_views = texts_z_views.view(T, C, -1)
        texts_z_views = texts_z_views / texts_z_views.norm(dim=-1, keepdim=True)
        self._texts_z_views = texts_z_views

        return self._texts_z_views

    def setup(self, stage: str) -> None:
        """Setup the model.

        Args:
            stage (str): Stage of the model.
        """
        super().setup(stage)

        # init metrics
        self.metrics = torch.nn.ModuleDict()
        acc_kwargs = {"task": "multiclass", "num_classes": len(self.trainer.datamodule.classes)}

        # loss metrics
        self.metrics["test/loss"] = MeanMetric()

        # classification metrics
        self.metrics["test/acc"] = MetricCollection(
            {f"test/acc@{k}": Accuracy(**acc_kwargs, top_k=k) for k in (1, 3, 5)}
        )

        # vocabulary metrics
        self.metrics["test/vocab_size"] = MeanMetric()
        self.metrics["test/unique_candidates"] = UniqueValues()
        self.metrics["test/unique_names"] = UniqueValues()

        # semantic metrics
        semantic_metrics = {}
        semantic_metrics["test/semantic_cluster_acc"] = SemanticClusterAccuracy()
        semantic_metrics["test/semantic_iou"] = SemanticIOU()
        semantic_metrics["test/semantic_similarity"] = SentenceScore()
        self.metrics["test/semantic_metrics"] = MetricCollection(semantic_metrics)

    def batch_step(self, images: torch.Tensor, texts_z: torch.Tensor) -> tuple:
        """Perform a single batch step.

        Args:
            images (torch.Tensor): Batch of images.
            texts_z (torch.Tensor): Batch of text embeddings.
        """
        images_z = self.vision_encoder(images)
        images_p = self.classifier(images_z, texts_z)
        texts_p = images_p.t()

        return images_p, texts_p, images_z, texts_z

    def test_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        images, targets = batch[0], batch[1]

        # compute loss
        images_p, _, _, _ = self.batch_step(images, self.texts_z_views)
        loss = F.nll_loss(torch.log(images_p), targets, reduction="mean")

        # log metrics
        self.metrics["test/loss"](loss)
        self.log("test/loss", self.metrics["test/loss"], prog_bar=True)
        self.metrics["test/acc"](images_p, targets)
        self.log_dict(self.metrics["test/acc"], prog_bar=True)

        # log vocabulary metrics
        vocabs = self.texts_views
        preds = images_p.argmax(dim=-1)
        words = [self.trainer.datamodule.classes[p] for p in preds]
        self.metrics["test/vocab_size"](sum(len(vocab) for vocab in vocabs))
        self.log("test/vocab_size", self.metrics["test/vocab_size"], prog_bar=True)
        self.metrics["test/unique_candidates"](vocabs)
        self.log("test/unique_candidates", self.metrics["test/unique_candidates"], prog_bar=True)
        self.metrics["test/unique_names"](words)
        self.log("test/unique_names", self.metrics["test/unique_names"], prog_bar=True)

        # store outputs for later evaluation
        semantic_targets = [self.trainer.datamodule.classes[t] for t in targets]
        self.test_outputs.append((words, semantic_targets))

    def on_test_epoch_end(self):
        # log semantic metrics
        words, semantic_targets = zip(*self.test_outputs)
        words = sum(words, [])
        semantic_targets = sum(semantic_targets, [])
        self.metrics["test/semantic_metrics"](words, semantic_targets)
        self.log_dict(self.metrics["test/semantic_metrics"])

        super().on_test_epoch_end()

    @property
    def learnable_params(self) -> list[dict[str, Any]]:
        """Defines learnable parameters of the model."""
        return [{}]


if __name__ == "__main__":
    _ = CLIP()
