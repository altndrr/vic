from typing import Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric
from torchmetrics.aggregation import BaseAggregator
from transformers import AutoModel, AutoTokenizer

__all__ = [
    "SemanticClusterAccuracy",
    "SemanticIOU",
    "SentenceScore",
    "UniqueValues",
]


class SemanticClusterAccuracy(Metric):
    """Metric to evaluate the semantic clustering accuracy.

    It takes as input a list of predicted words and a list of target words. The metric cluster
    samples according to their predicted words and solves the optimal assignment problem to find
    the best mapping between the predicted clusters and the target clusters. The metric then
    computes the accuracy as the number of samples correctly assigned to their target cluster
    divided by the total number of samples.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.targets = []
        self.preds = []

    def update(self, preds: list[str], targets: list[str]):
        """Update state with data.

        Args:
            preds (list[str]): Predicted words.
            targets (list[str]): Targets words.
        """
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        targets = sum(self.targets, [])
        preds = sum(self.preds, [])

        # assign a unique idx to each word
        unique_words = list(set(preds))
        words_to_idx = {w: i for i, w in enumerate(unique_words)}
        preds = np.array([words_to_idx[w] for w in preds])

        # assign a unique idx to each target
        unique_targets = list(set(targets))
        targets_to_idx = {w: i for i, w in enumerate(unique_targets)}
        targets = np.array([targets_to_idx[w] for w in targets])

        targets -= targets.min()
        mapping, w = self._compute_best_mapping(targets, preds)
        cluster_acc = sum([w[i, j] for i, j in mapping]) * 1.0 / preds.size

        return torch.tensor(cluster_acc, device=self.device)

    def _compute_best_mapping(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Compute the best mapping between predicted and target clusters.

        Args:
            y_true (np.ndarray): Target cluster idxs.
            y_pred (np.ndarray): Predicted clusters idxs.
        """
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        return np.transpose(np.asarray(linear_sum_assignment(w.max() - w))), w


class SemanticIOU(Metric):
    """Metric to evaluate the semantic intersection over union.

    It takes as input a list of predicted words and a list of target words. The metric computes the
    intersection and union of the predicted words and the target words and returns the intersection
    over union.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("intersection", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, values: list[str], targets: list[str]) -> None:
        """Update state with data.

        Args:
            values (list[str]): Predicted words.
            targets (list[str]): Targets words.
        """
        intersections = []
        unions = []
        for value, target in zip(values, targets):
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

    It takes as input a list of predicted sentences and a list of target sentences. The metric
    computes the cosine similarity between the embeddings of the predicted sentences and the
    target sentences.

    Args:
        model_name (str): Name of the model to use. Defaults to
            "sentence-transformers/all-MiniLM-L6-v2".
        cache_dir (str): Path to BERT cache directory. Defaults to ".cache".
    """

    def __init__(
        self,
        *args,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = ".cache",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self._model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)

        self.add_state("similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def encode(self, sentence: str, **kwargs) -> torch.Tensor:
        """Encode the input sentence with the BERT model.

        Args:
            sentence (str): Input sentence.
        """
        tokens = self._tokenizer.encode_plus(
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

    def update(self, values: list[str], targets: list[str]) -> None:
        """Update state with data.

        Args:
            values (list[str]): Predicted sentences.
            targets (list[str]): Target sentences.
        """
        values_z = []
        targets_z = []
        for value, target in zip(values, targets):
            values_z.append(self.encode(value))
            targets_z.append(self.encode(target))

        values_z = torch.stack(values_z)
        targets_z = torch.stack(targets_z)

        # compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(values_z, targets_z)

        self.similarity += torch.sum(similarity)
        self.total += len(values)

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        return self.similarity.float() / self.total


class UniqueValues(BaseAggregator):
    """Metric that counts the number of unique values.

    Args:
        nan_strategy: Strategy to handle NaN values. Can be `error`, `warn`, or `ignore`.
            Defaults to `warn`.
    """

    full_state_update = True

    def __init__(self, nan_strategy: Union[str, float] = "warn", **kwargs):
        super().__init__(None, -torch.tensor(0), nan_strategy, **kwargs)
        self._unique_values = set()

    def update(self, value: Union[list[str], list[list[str]]]) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
        """
        if isinstance(value, list) and isinstance(value[0], list):
            value = sum(value, [])
        self._unique_values.update(value)
        self.value = torch.tensor(len(self._unique_values))
