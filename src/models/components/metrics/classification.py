from collections import defaultdict
from typing import Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric


class SemanticClusterAccuracy(Metric):
    """Metric to evaluate the semantic clustering accuracy.

    It takes as input a batch of predicted words and a batch of target words. The metric cluster
    samples according to their predicted words and solves the optimal assignment problem to find
    the best mapping between the predicted clusters and the target clusters. The metric then
    computes the accuracy as the number of samples correctly assigned to their target cluster
    divided by the total number of samples.

    Args:
        task (str): Task to perform. Either "multiclass" or "multilabel".
        average (str): Type of averaging to perform. Either "micro" or "macro". "micro" computes
            the metric globally, while "macro" computes the metric for each class and then takes
            the average.
    """

    def __init__(self, *args, task: str = "multilabel", average: str = "micro", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert task in ["multiclass", "multilabel"]
        assert average in ["micro", "macro"]
        self.average = average
        self.task = task
        self.targets = []
        self.values = []
        self.targets_per_sample = []
        self.values_per_sample = []

    def update(self, values: list[dict], targets: Union[list[str], list[list[str]]]) -> None:
        """Update state with data.

        Args:
            values (list[dict]): Predicted words.
            targets (list[str] | list[list[str]]): Targets words.
        """
        if isinstance(targets, list) and not isinstance(targets[0], list):
            targets = [[t] for t in targets]

        if self.task == "multiclass":
            values = [[max(v, key=v.get)] for v in values]  # select the most probable word
        elif self.task == "multilabel":
            values = [list(v.keys()) for v in values]  # convert the values to its list of keys

        values_per_sample = [len(v) for v in values]
        targets_per_sample = [len(t) for t in targets]

        if self.task == "multiclass":
            assert all([v == 1 for v in values_per_sample])
            assert all([t == 1 for t in targets_per_sample])

        self.values.append(values)
        self.targets.append(targets)
        self.values_per_sample.append(values_per_sample)
        self.targets_per_sample.append(targets_per_sample)

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        targets = sum(self.targets, [])
        values = sum(self.values, [])
        targets_per_sample = sum(self.targets_per_sample, [])
        values_per_sample = sum(self.values_per_sample, [])

        # expand the values and targets to have all their combinations for the same sample
        values = sum([v * t for v, t in zip(values, targets_per_sample)], [])
        targets = sum([t * v for t, v in zip(targets, values_per_sample)], [])

        # assign a unique idx to each word
        unique_words = list(set(values))
        words_to_idx = {w: i for i, w in enumerate(unique_words)}
        values = np.array([words_to_idx[w] for w in values])

        # assign a unique idx to each target
        unique_targets = list(set(targets))
        targets_to_idx = {w: i for i, w in enumerate(unique_targets)}
        targets = np.array([targets_to_idx[w] for w in targets])

        # map the targets to the values
        targets -= targets.min()
        mapping, w = self._compute_best_mapping(targets, values)
        mapping = [(i, j) for i, j in mapping if j < len(unique_targets)]
        targets_score = defaultdict(int)
        for i, j in mapping:
            targets_score[unique_targets[j]] += w[i, j]

        if self.average == "micro":
            cluster_acc = sum(targets_score.values()) / values.size
        elif self.average == "macro":
            targets_count = np.unique(targets, return_counts=True)
            targets_count = {unique_targets[j]: c for j, c in zip(*targets_count)}
            targets_acc = [targets_score[t] / targets_count[t] for t in targets_score.keys()]
            cluster_acc = sum(targets_acc) / len(targets_acc)

        return torch.tensor(cluster_acc, device=self.device)

    def _compute_best_mapping(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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
        mapping = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
        return mapping, w
