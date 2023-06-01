import torch

from src import utils
from src.data.components.transforms import TextCompose, default_vocabulary_transforms
from src.models.vocabulary_free_clip import VocabularyFreeCLIP

log = utils.get_logger(__name__)


class CaSED(VocabularyFreeCLIP):
    """LightningModule for Category Search from External Databases.

    Args:
        vocabulary (BaseVocabulary): Vocabulary to use.
        vocabulary_transforms (TextCompose): List of transforms to apply to the
            vocabulary.
        model_name (str): Name of the CLIP model to use.
        pretrained (str): Pretrained weights to use for the CLIP model. Defaults to "openai".

    Extra hparams:
        alpha (float): Weight for the average of the image and text predictions. Defaults to 0.5.
        vocabulary_prompt (str): Prompt to use for the vocabulary. Defaults to "{}".
        tau (float): Temperature to use for the classifier. Defaults to 1.0.
    """

    def __init__(
        self, *args, vocabulary_transforms: TextCompose = default_vocabulary_transforms(), **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocabulary_transforms = vocabulary_transforms

        # save hyperparameters
        kwargs["alpha"] = kwargs.get("alpha", 0.5)
        self.save_hyperparameters("alpha", "vocabulary_transforms")

    def batch_step(self, images_z: torch.Tensor, vocabularies: list[list]) -> list:
        """Perform a single batch step.

        Args:
            images_z (torch.Tensor): Batch of image embeddings.
            images_fp (list[str]): List of paths to image files.
        """
        words = []

        alpha = self.hparams.alpha

        for image_z, vocabulary in zip(images_z, vocabularies):
            image_z = image_z.unsqueeze(0)

            # generate a single text embedding from the unfiltered vocabulary
            unfiltered_vocabulary_z = self.encode_vocabulary(vocabulary).squeeze(0)
            text_z = unfiltered_vocabulary_z.mean(dim=0)
            text_z = text_z / text_z.norm(dim=-1, keepdim=True)
            text_z = text_z.unsqueeze(0)

            # filter the vocabulary, embed it, and get its mean embedding
            vocabulary = self.vocabulary_transforms(vocabulary) or ["object"]
            vocabulary_z = self.encode_vocabulary(vocabulary, use_prompts=True)
            mean_vocabulary_z = vocabulary_z.mean(dim=0)
            mean_vocabulary_z = mean_vocabulary_z / mean_vocabulary_z.norm(dim=-1, keepdim=True)

            # get the image and text predictions
            image_p = self.classifier(image_z, vocabulary_z)
            text_p = self.classifier(text_z, vocabulary_z)

            # average the image and text predictions
            sample_p = alpha * image_p + (1 - alpha) * text_p
            pred = sample_p.argmax(dim=-1)
            words.append(vocabulary[pred.item()])

        return words


if __name__ == "__main__":
    _ = CaSED()
