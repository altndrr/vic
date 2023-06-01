from src.models.components.nn.classifiers import NearestNeighboursClassifier
from src.models.components.nn.encoders import LanguageTransformer

__all__ = ["NearestNeighboursClassifier", "LanguageTransformer"]

CLASSIFIERS = {
    "nearest_neighbours": NearestNeighboursClassifier,
}

LANGUAGE_ENCODERS = {
    "transformer": LanguageTransformer,
}
