from src.models.components.metrics.aggregation import UniqueValues
from src.models.components.metrics.classification import SemanticClusterAccuracy
from src.models.components.metrics.text import SentenceIOU, SentenceScore

__all__ = [
    "SemanticClusterAccuracy",
    "SemanticJaccardIndex",
    "SemanticRecall",
    "SentenceScore",
    "SentenceIOU",
    "UniqueValues",
]

AGGREGATION = {
    "unique_values": UniqueValues,
}

CLASSIFICATION = {
    "semantic_cluster_accuracy": SemanticClusterAccuracy,
}

TEXT = {
    "sentence_iou": SentenceIOU,
    "sentence_score": SentenceScore,
}
