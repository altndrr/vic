import torch

from src import utils
from src.models.clip import CLIP
from src.models.components.vocabularies import BaseVocabulary, ImageNetVocabulary

log = utils.get_logger(__name__)


class VocabularyFreeCLIP(CLIP):
    """LightningModule for Contrastive Language-Image Pre-training without a vocabulary.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".

    Extra hparams:
        vocabulary_prompt (str): Prompt to use for the vocabulary. Defaults to "{}".
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    def __init__(self, *args, vocabulary: BaseVocabulary = ImageNetVocabulary(), **kwargs) -> None:
        super().__init__(*args, prompt="{}", **kwargs)
        self.vocabulary = vocabulary
        self._prev_vocab_words = None
        self._prev_used_prompts = None
        self._prev_vocab_words_z = None

        # save hyperparameters
        vocabulary_prompt = kwargs.get("vocabulary_prompt", "{}")
        kwargs["vocabulary_prompts"] = [vocabulary_prompt]
        self.save_hyperparameters("vocabulary_prompts")

    def encode_vocabulary(self, vocabulary: list, use_prompts: bool = False) -> torch.Tensor:
        """Encode a vocabulary.

        Args:
            vocabulary (list): List of words.
        """
        # check if vocabulary has changed
        if vocabulary == self._prev_vocab_words and use_prompts == self._prev_used_prompts:
            return self._prev_vocab_words_z

        # tokenize vocabulary
        classes = [c.replace("_", " ") for c in vocabulary]
        prompts = self.hparams.vocabulary_prompts if use_prompts else ["{}"]
        texts_views = [[p.format(c) for c in classes] for p in prompts]
        tokenized_texts_views = [
            torch.cat([self.tokenizer(prompt) for prompt in class_prompts])
            for class_prompts in texts_views
        ]
        tokenized_texts_views = torch.stack(tokenized_texts_views).to(self.device)

        # encode vocabulary
        T, C, _ = tokenized_texts_views.shape
        texts_z_views = self.language_encoder(tokenized_texts_views.view(T * C, -1))
        texts_z_views = texts_z_views.view(T, C, -1)
        texts_z_views = texts_z_views / texts_z_views.norm(dim=-1, keepdim=True)

        # cache vocabulary
        self._prev_vocab_words = vocabulary
        self._prev_used_prompts = use_prompts
        self._prev_vocab_words_z = texts_z_views

        return texts_z_views

    def batch_step(self, images_z: torch.Tensor, vocabularies: list[list]) -> list:
        """Perform a single batch step.

        Args:
            images_z (torch.Tensor): Batch of image embeddings.
            images_fp (list[str]): List of paths to image files.
        """
        words = []

        for image_z, vocabulary in zip(images_z, vocabularies):
            image_z = image_z.unsqueeze(0)

            # embed the vocabulary and remove the prompt dimension
            vocabulary_z = self.encode_vocabulary(vocabulary).squeeze(0)

            # get the image prediction
            image_p = self.classifier(image_z, vocabulary_z)
            pred = image_p.argmax(dim=-1)
            words.append(vocabulary[pred.item()])

        return words

    def test_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        images, targets, images_fp = batch[0], batch[1], batch[2]
        images_z = self.vision_encoder(images)
        vocabularies = self.vocabulary(images_z=images_z, images_fp=images_fp)

        words = self.batch_step(images_z, vocabularies)

        # log vocabulary metrics
        vocabs = vocabularies
        len_vocabularies = torch.tensor([len(v) for v in vocabularies]).to(self.device)
        self.metrics["test/vocab_size"](len_vocabularies)
        self.log("test/vocab_size", self.metrics["test/vocab_size"])
        self.metrics["test/unique_candidates"](vocabs)
        self.log("test/unique_candidates", self.metrics["test/unique_candidates"])
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


if __name__ == "__main__":
    _ = VocabularyFreeCLIP()
