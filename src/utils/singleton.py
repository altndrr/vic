from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


class SentenceBERT:
    """Singleton model of SentenceBERT model for sentence embeddings.

    Attributes:
        model_name (str): Name of the model to use. Defaults to
            "sentence-transformers/all-MiniLM-L6-v2".
        cache_dir (str): Path to the cache directory. Defaults to ".cache".
    """

    _instance = None

    def __new__(cls, model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir=".cache"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registered_names = []
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._instance.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            cls._instance.model = cls._instance.model.to(cls._instance.device)
        return cls._instance

    def __call__(self, sentence: list[str], **kwargs) -> torch.Tensor:
        """Encode the input sentence with the BERT model.

        Args:
            sentence (list[str]): Input sentences.
        """
        assert self.tokenizer is not None

        tokens = self.tokenizer(
            sentence, max_length=128, truncation=True, padding="max_length", return_tensors="pt"
        ).to(self.device)
        embeddings = self.model(**tokens).last_hidden_state

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

    def __getattr__(self, name: str) -> Any:
        """Get the attribute from the model.

        Args:
            name (str): Name of the attribute.
        """
        if name not in self._registered_names:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        return getattr(self.model, name)

    def register_buffer(self, name: str, buffer: torch.Tensor, exists_ok: bool = False) -> None:
        """Register a buffer with the model.

        Args:
            name (str): Name of the buffer.
            buffer (torch.Tensor): Buffer to register.
            exists_ok (bool, optional): Whether to allow overwriting existing buffers.
                Defaults to False.
        """
        if hasattr(self.model, name):
            if not exists_ok:
                raise ValueError(f"Buffer with name '{name}' already exists.")
            return

        self._registered_names.append(name)
        self.model.register_buffer(name, buffer)
