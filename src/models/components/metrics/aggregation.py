from typing import Union

import torch
from torchmetrics.aggregation import BaseAggregator


class UniqueValues(BaseAggregator):
    """Metric that counts the number of unique values.

    Args:
        nan_strategy: Strategy to handle NaN values. Can be `error`, `warn`, or `ignore`.
            Defaults to `warn`.
    """

    full_state_update = True

    def __init__(self, nan_strategy: Union[str, float] = "warn", **kwargs) -> None:
        super().__init__(None, -torch.tensor(0), nan_strategy, **kwargs)
        self.unique_values = set()

    def update(self, value: Union[list[str], list[list[str]]]) -> None:
        """Update state with data.

        Args:
            value (list[str] | list[list[str]]): Either a float or tensor containing data.
                Additional tensor dimensions will be flattened
        """
        if isinstance(value, list) and isinstance(value[0], list):
            value = sum(value, [])
        self.unique_values.update(value)
        self.value = torch.tensor(len(self.unique_values))
