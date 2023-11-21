from typing import Union

import torch
from torchmetrics import Metric

from src import utils


class SentenceIOU(Metric):
    """Metric to evaluate the intersection over union of words between two sentences.

    It takes as input a batch of predicted words with their scores and a batch of target sentences.
    The metric computes the intersection and union of the most probable predicted words (i.e.,
    top-1) and the target words and returns the intersection over union.

    Args:
        task (str): Task to perform. Currently only "multiclass" is supported.
    """

    def __init__(self, *args, task: str = "multiclass", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert task in ["multiclass"]
        self.task = task
        self.add_state("intersection", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, values: list[dict], targets: Union[list[str], list[list[str]]]) -> None:
        """Update state with data.

        Args:
            values (list[dict]): Predicted words.
            targets (list[str] | list[list[str]]): Targets words.
        """
        if isinstance(targets, list) and isinstance(targets[0], list):
            assert len(targets[0]) == 1, "Only one target per sample is supported."
            targets = sum(targets, [])

        intersections = []
        unions = []
        for value, target in zip(values, targets):
            value = max(value, key=value.get)  # take the word with the highest score
            value = "".join([c for c in value if c.isalnum() or c == " "]).lower()
            target = "".join([c for c in target if c.isalnum() or c == " "]).lower()

            intersections.append(len(set(value.split()) & set(target.split())))
            unions.append(len(set(value.split()) | set(target.split())))

        intersections = torch.tensor(intersections, device=self.device)
        unions = torch.tensor(unions, device=self.device)

        self.intersection = torch.cat([self.intersection, intersections])
        self.union = torch.cat([self.union, unions])

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        return torch.mean(self.intersection.float() / self.union.float())


class SentenceScore(Metric):
    """Metric to evaluate the similarity between two sentences.

    It takes as input a batch of predicted sentences with their scores and a batch of target
    sentences. The metric computes the cosine similarity between the embeddings of the most
    probable sentences (i.e., top-1) and the target sentences.

    Args:
        task (str): Task to perform. Currently only "multiclass" is supported.
    """

    def __init__(self, *args, task: str = "multiclass", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert task in ["multiclass"]
        self.encoder = utils.SentenceBERT()
        self.task = task
        self.add_state("similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def encode(self, sentence: list[str], **kwargs) -> torch.Tensor:
        """Encode the input sentence with the BERT model.

        Args:
            sentence (list[str]): Input sentences.
        """
        tokens = self._tokenizer(
            sentence, max_length=128, truncation=True, padding="max_length", return_tensors="pt"
        ).to(self.device)
        embeddings = self._model(**tokens).last_hidden_state

        # mask out padding tokens
        mask = tokens["attention_mask"].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask

        # sum over all tokens
        summed = torch.sum(masked_embeddings, dim=1)
        summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        # normalise and remove batch dimension
        embeddings = summed / summed_mask
        embeddings = embeddings.squeeze(0)

        return embeddings

    def update(self, values: list[dict], targets: Union[list[str], list[list[str]]]) -> None:
        """Update state with data.

        Args:
            values (list[dict]): Predicted sentences.
            targets (list[str] | list[list[str]]): Target sentences.
        """
        if isinstance(targets, list) and isinstance(targets[0], list):
            assert len(targets[0]) == 1, "Only one target per sample is supported."
            targets = sum(targets, [])

        values_z = self.encoder([max(value, key=value.get) for value in values])
        targets_z = self.encoder(targets)

        # compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(values_z, targets_z)

        self.similarity += torch.sum(similarity)
        self.total += len(values)

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        return self.similarity.float() / self.total
