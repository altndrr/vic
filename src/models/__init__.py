from src.models.cased import CaSED
from src.models.clip import CLIP
from src.models.vocabulary_free_clip import VocabularyFreeCLIP

__all__ = ["CaSED", "CLIP", "VocabularyFreeCLIP"]

MODELS = {
    "cased": CaSED,
    "clip": CLIP,
    "vocabulary_free_clip": VocabularyFreeCLIP,
}
