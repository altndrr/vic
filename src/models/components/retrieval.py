import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

__all__ = ["RetrievalClient", "RetrievalResult"]


@dataclass
class RetrievalResult:
    """Result of a query to the retrieval system.

    Args:
        id (str): id of the image.
        caption (str): caption of the image.
        similarity (float): similarity score.
        sample_z (list): embedding of the retrieved samples.
        image_path (str, optional): path of the image.
    """

    id: str
    caption: str
    similarity: float
    sample_z: list
    image_path: Optional[str] = None

    def __str__(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return self.__str__()


class RetrievalClient:
    """Client for querying the retrieval server.

    Args:
        url (str): URL of the backend.
        index_name (str): which index to search over e.g. "laion5B" or "laion_400m".
        is_multilingual (bool): use a multilingual version of clip. Default is False.
        aesthetic_score (int): ranking score for aesthetic, higher is prettier. Default is 9.
        aesthetic_weight (float): weight of the aesthetic score, between 0 and 1. Default is 0.5.
        modality (Modality): Search modality. One of Modality.IMAGE or Modality.TEXT. Default is
            Modality.IMAGE.
        num_samples (int): Number of samples to return. Default is 40.
        deduplicate (bool): Whether to deduplicate the result by image embedding. Default is true.
        use_safety_model (bool): Whether to remove unsafe images. Default is true.
        use_violence_detector (bool): Whether to remove images with violence. Default is true.
    """

    def __init__(
        self,
        url: str,
        index_name: str,
        is_multilingual: bool = False,
        aesthetic_score: int = 9,
        aesthetic_weight: float = 0.5,
        modality: str = "text",
        num_samples: int = 40,
        deduplicate: bool = True,
        use_safety_model: bool = True,
        use_violence_detector: bool = True,
    ):
        assert modality in ["image", "text"], "Modality must be one of `image` or `text`"

        self._url = url
        self._index_name = index_name
        self._is_multilingual = is_multilingual
        self._aesthetic_score = aesthetic_score
        self._aesthetic_weight = aesthetic_weight
        self._modality = modality
        self._num_samples = num_samples
        self._deduplicate = deduplicate
        self._use_safety_model = use_safety_model
        self._use_violence_detector = use_violence_detector

    def query(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        embedding: Optional[list] = None,
    ) -> list[list[RetrievalResult]]:
        """Search for similar images or captions given a text or image input.

        Args:
            text: text to be searched semantically.
            image: base64 string of image to be searched semantically
            embedding: embedding representation of the input

        Returns:
            list of RetrievalResult objects.
        """
        assert text or image or embedding, "Either text or image or embedding must be provided."

        if text:
            return self.__search_knn_api__(text=text)

        if image and image.startswith("http"):
            return self.__search_knn_api__(image_url=image)

        if image:
            assert Path(image).exists(), f"{image} does not exist."
            return self.__search_knn_api__(image=image)

        if embedding:
            return self.__search_knn_api__(embedding=embedding)

        raise ValueError("Either text or image or embedding must be provided.")

    def __search_knn_api__(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        image_url: Optional[str] = None,
        embedding: Optional[list] = None,
    ) -> list[list[RetrievalResult]]:
        """Send a request to the knn service.

        It represents a direct API call and should not be called directly outside the package.

        Args:
            text: text to be searched semantically.
            image: base64 string of image to be searched semantically
            image_url: url of the image to be searched semantically
            embedding: embedding representation of the input

        Returns:
            list of RetrievalResult objects.
        """
        if image:
            with open(image, "rb") as f:
                encoded_string = base64.b64encode(f.read())
                image = str(encoded_string.decode("utf-8"))

        payload = {
            "text": text,
            "image": image,
            "image_url": image_url,
            "embedding_input": embedding,
            "deduplicate": self._deduplicate,
            "use_safety_model": self._use_safety_model,
            "use_violence_detector": self._use_violence_detector,
            "indice_name": self._index_name,
            "use_mclip": self._is_multilingual,
            "aesthetic_score": self._aesthetic_score,
            "aesthetic_weight": self._aesthetic_weight,
            "modality": self._modality,
            "num_images": self._num_samples,
            "num_result_ids": self._num_samples,
        }

        try:
            response = requests.post(self._url, data=json.dumps(payload), timeout=30.0).json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Could not connect to the retrieval server. Did you start it? "
                "To start the server, open a terminal and run `docker compose build` "
                "to build the docker images and then "
                "`docker compose --profile retrieval-server up` to start the server.\n"
                "Refer to the README for more information."
            )

        formatted_response = []
        for res in response:
            res = [RetrievalResult(**r) for r in res]
            formatted_response.append(res)

        return formatted_response
