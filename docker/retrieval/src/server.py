import base64
import math
from io import BytesIO
from typing import Any, Optional, Union

import faiss
import numpy as np
from clip_retrieval.clip_back import ClipOptions, ClipResource, IndicesList
from clip_retrieval.clip_back import KnnService as _KnnService
from clip_retrieval.clip_back import (
    MetadataService,
    MetricsSummary,
    convert_metadata_to_base64,
    download_image,
    load_clip_indices,
    meta_to_dict,
    normalized,
)
from flask import Flask
from flask_cors import CORS
from flask_restful import Api
from PIL import Image
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware


class KnnService(_KnnService):
    """KNN Service to retrieve the nearest neighbors from image or text."""

    def compute_query(
        self,
        clip_resource: ClipResource,
        text_input: Optional[Union[str, list[str]]] = None,
        image_input: Optional[Union[str, list[str]]] = None,
        image_url_input: Optional[Union[str, list[str]]] = None,
        embedding_input: Optional[Union[list[float], list[list[float]]]] = None,
        use_mclip: bool = False,
        aesthetic_score: int = 9,
        aesthetic_weight: float = 0.5,
    ) -> np.matrix:
        """Compute the query.

        Args:
            clip_resource (ClipResource): CLIP resource.
            text_input (str | list[str], optional): Text input. Defaults to `None`.
            image_input (str | list[str], optional): Image input. Defaults to `None`.
            image_url_input (str | list[str], optional): Image URL input. Defaults to `None`.
            embedding_input (list[float] | list[list[float]], optional): Embedding input. Defaults
                to `None`.
            use_mclip (bool, optional): Whether to use mclip. Defaults to `False`.
            aesthetic_score (int): Aesthetic score. Defaults to `9`.
            aesthetic_weight (float): Aesthetic weight. Defaults to `0.5`.
        """
        import clip
        import torch

        if isinstance(text_input, str):
            text_input = [text_input]
        if isinstance(image_input, str):
            image_input = [image_input]
        if isinstance(image_url_input, str):
            image_url_input = [image_url_input]
        if isinstance(embedding_input, list) and isinstance(embedding_input[0], float):
            embedding_input = [embedding_input]

        is_text_valid = text_input is not None and len(text_input) > 0 and text_input[0] != ""
        is_image_valid = image_input is not None and len(image_input) > 0
        is_image_url_valid = image_url_input is not None and len(image_url_input) > 0
        is_embedding_valid = embedding_input is not None and len(embedding_input) > 0

        if is_text_valid:
            if use_mclip:
                query = [normalized(clip_resource.model_txt_mclip(t)) for t in text_input]
                query = np.matrix(query)
            else:
                text = clip.tokenize(text_input, truncate=True).to(clip_resource.device)
                with torch.no_grad():
                    text_features = clip_resource.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                query = text_features.cpu().to(torch.float32).detach().numpy()
        elif is_image_valid or is_image_url_valid:
            if image_input is not None:
                img_data = [BytesIO(base64.b64decode(image)) for image in image_input]
            elif image_url_input is not None:
                img_data = [download_image(image) for image in image_url_input]
            img = [Image.open(i) for i in img_data]
            prepro = torch.stack([clip_resource.preprocess(i) for i in img])
            prepro = prepro.to(clip_resource.device)
            with torch.no_grad():
                image_features = clip_resource.model.encode_image(prepro)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            query = image_features.cpu().to(torch.float32).detach().numpy()
        elif is_embedding_valid:
            query = np.matrix(embedding_input).astype("float32")

        if clip_resource.aesthetic_embeddings is not None and aesthetic_score is not None:
            aesthetic_embedding = clip_resource.aesthetic_embeddings[aesthetic_score]
            query = query + aesthetic_embedding * aesthetic_weight
            query = query / np.linalg.norm(query)

        return query

    def knn_search(
        self,
        query: np.matrix,
        modality: str,
        num_result_ids: int,
        clip_resource: ClipResource,
        deduplicate: bool,
        use_safety_model: bool,
        use_violence_detector: bool,
    ):
        """Perform KNN search.

        Args:
            query (np.matrix): Query to search.
            modality (str): Modality to search.
            num_result_ids (int): Number of result ids.
            clip_resource (ClipResource): CLIP resource.
            deduplicate (bool): Whether to deduplicate results.
            use_safety_model (bool): Whether to use safety model.
            use_violence_detector (bool): Whether to use violence detector.
        """

        image_index = clip_resource.image_index
        text_index = clip_resource.text_index
        if clip_resource.metadata_is_ordered_by_ivf:
            ivf_mapping = clip_resource.ivf_old_to_new_mapping

        index = image_index if modality == "image" else text_index

        if clip_resource.metadata_is_ordered_by_ivf:
            previous_nprobe = faiss.extract_index_ivf(index).nprobe
            if num_result_ids >= 100000:
                nprobe = math.ceil(num_result_ids / 3000)
                params = faiss.ParameterSpace()
                params.set_index_parameters(
                    index, f"nprobe={nprobe},efSearch={nprobe*2},ht={2048}"
                )
        distances, indices, embeddings = index.search_and_reconstruct(query, num_result_ids)
        if clip_resource.metadata_is_ordered_by_ivf:
            results = [np.take(ivf_mapping, indices[i]) for i in range(len(indices))]
        else:
            results = [indices[i] for i in range(len(indices))]
        if clip_resource.metadata_is_ordered_by_ivf:
            params = faiss.ParameterSpace()
            params.set_index_parameters(
                index, f"nprobe={previous_nprobe},efSearch={previous_nprobe*2},ht={2048}"
            )

        nb_results = [np.where(r == -1)[0] for r in results]
        total_distances = []
        total_indices = []
        total_embeddings = []
        for i in range(len(results)):
            num_res = nb_results[i][0] if len(nb_results[i]) > 0 else len(results[i])

            result_indices = results[i][:num_res]
            result_distances = distances[i][:num_res]
            result_embeddings = embeddings[i][:num_res]
            result_embeddings = normalized(result_embeddings)
            local_indices_to_remove = self.post_filter(
                clip_resource.safety_model,
                result_embeddings,
                deduplicate,
                use_safety_model,
                use_violence_detector,
                clip_resource.violence_detector,
            )

            indices_to_remove = set()
            for local_index in local_indices_to_remove:
                indices_to_remove.add(result_indices[local_index])

            curr_indices = []
            curr_distances = []
            curr_embeddings = []
            for ind, dis, emb in zip(result_indices, result_distances, result_embeddings):
                if ind not in indices_to_remove:
                    indices_to_remove.add(ind)
                    curr_indices.append(ind)
                    curr_distances.append(dis)
                    curr_embeddings.append(emb)

            total_indices.append(curr_indices)
            total_distances.append(curr_distances)
            total_embeddings.append(curr_embeddings)

        return total_distances, total_indices, total_embeddings

    def query(
        self,
        text_input: Optional[str] = None,
        image_input: Optional[str] = None,
        image_url_input: Optional[str] = None,
        embedding_input: Optional[list] = None,
        modality: str = "image",
        num_images: int = 100,
        num_result_ids: int = 100,
        indice_name: Optional[str] = None,
        use_mclip: bool = False,
        deduplicate: bool = True,
        use_safety_model: bool = False,
        use_violence_detector: bool = False,
        aesthetic_score: Optional[int] = None,
        aesthetic_weight: Optional[float] = None,
    ):
        """Implement the querying functionality of the knn service.

        Add function to query from embedding input: from text and image to nearest neighbors.

        Args:
            text_input (str, optional): Text input. Defaults to `None`.
            image_input (str, optional): Image input. Defaults to `None`.
            image_url_input (str, optional): Image url input. Defaults to `None`.
            embedding_input (list, optional): Embedding input. Defaults to `None`.
            modality (str): Modality to search. Defaults to `image`.
            num_images (int): Number of images to search. Defaults to `100`.
            num_result_ids (int): Number of result ids. Defaults to `100`.
            indice_name (str): Indice name. Defaults to `None`.
            use_mclip (bool): Whether to use multi-lingual clip. Defaults to `False`.
            deduplicate (bool): Whether to deduplicate results. Defaults to `True`.
            use_safety_model (bool): Whether to use safety model. Defaults to `False`.
            use_violence_detector (bool): Whether to use violence detector. Defaults to `False`.
            aesthetic_score (float): Aesthetic score. Defaults to `None`.
            aesthetic_weight (float): Aesthetic weight. Defaults to `None`.
        """

        if not text_input and not image_input and not image_url_input and not embedding_input:
            raise ValueError("must fill one of text, image and image url input")
        if indice_name is None:
            indice_name = next(iter(self.clip_resources.keys()))

        clip_resource = self.clip_resources[indice_name]

        query = self.compute_query(
            clip_resource=clip_resource,
            text_input=text_input,
            image_input=image_input,
            image_url_input=image_url_input,
            embedding_input=embedding_input,
            use_mclip=use_mclip,
            aesthetic_score=aesthetic_score,
            aesthetic_weight=aesthetic_weight,
        )
        distances, indices, embs = self.knn_search(
            query,
            modality=modality,
            num_result_ids=num_result_ids,
            clip_resource=clip_resource,
            deduplicate=deduplicate,
            use_safety_model=use_safety_model,
            use_violence_detector=use_violence_detector,
        )
        if len(distances) == 0:
            return []

        total_results = []
        for i in range(len(distances)):
            results = self.map_to_metadata(
                indices[i],
                distances[i],
                embs[i],
                num_images,
                clip_resource.metadata_provider,
                clip_resource.columns_to_return,
            )
            total_results.append(results)

        return total_results

    def map_to_metadata(
        self,
        indices: list,
        distances: list,
        embs: list,
        num_images: int,
        metadata_provider: Any,
        columns_to_return: list,
    ):
        """Map the indices to metadata.

        Args:
            indices (list): List of indices.
            distances (list): List of distances.
            embs (list): List of results embeddings.
            num_images (int): Number of images.
            metadata_provider (Any): Metadata provider.
            columns_to_return (list): List of columns to return.
        """

        results = []
        metas = metadata_provider.get(indices[:num_images], columns_to_return)
        for key, (d, i, emb) in enumerate(zip(distances, indices, embs)):
            output = {}
            meta = None if key + 1 > len(metas) else metas[key]
            convert_metadata_to_base64(meta)
            if meta is not None:
                output.update(meta_to_dict(meta))
            output["id"] = i.item()
            output["similarity"] = d.item()
            output["sample_z"] = emb.tolist()
            results.append(output)

        return results


def clip_back(
    host: str,
    port: int = 1234,
    indices_paths: str = "indices_paths.json",
    enable_hdf5: bool = False,
    enable_faiss_memory_mapping: bool = False,
    columns_to_return: Optional[list] = None,
    reorder_metadata_by_ivf_index: bool = False,
    default_backend: Optional[str] = None,
    url_column: str = "url",
    enable_mclip_option: bool = True,
    clip_model: str = "ViT-B/32",
    use_jit: bool = True,
    use_arrow: bool = False,
    provide_safety_model: bool = False,
    provide_violence_detector: bool = False,
    provide_aesthetic_embeddings: bool = True,
):
    """Start the clip back service.

    Args:
        host (str): Host.
        port (int): Port.
        indices_paths (str): Path to indices paths.
        enable_hdf5 (bool): Whether to enable hdf5.
        enable_faiss_memory_mapping (bool): Whether to enable faiss memory mapping.
        columns_to_return (list): List of columns to return.
        reorder_metadata_by_ivf_index (bool): Whether to reorder metadata by ivf index.
        default_backend (str): Default backend.
        url_column (str): Url column.
        enable_mclip_option (bool): Whether to enable mclip option.
        clip_model (str): Clip model.
        use_jit (bool): Whether to use jit.
        use_arrow (bool): Whether to use arrow.
        provide_safety_model (bool): Whether to provide safety model.
        provide_violence_detector (bool): Whether to provide violence detector.
        provide_aesthetic_embeddings (bool): Whether to provide aesthetic embeddings.
    """
    print("starting boot of clip back")
    if columns_to_return is None:
        columns_to_return = ["image_path", "caption"]
    clip_resources = load_clip_indices(
        indices_paths=indices_paths,
        clip_options=ClipOptions(
            indice_folder="",
            clip_model=clip_model,
            enable_hdf5=enable_hdf5,
            enable_faiss_memory_mapping=enable_faiss_memory_mapping,
            columns_to_return=columns_to_return,
            reorder_metadata_by_ivf_index=reorder_metadata_by_ivf_index,
            enable_mclip_option=enable_mclip_option,
            use_jit=use_jit,
            use_arrow=use_arrow,
            provide_safety_model=provide_safety_model,
            provide_violence_detector=provide_violence_detector,
            provide_aesthetic_embeddings=provide_aesthetic_embeddings,
        ),
    )
    print("indices loaded")

    app = Flask(__name__)
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": make_wsgi_app()})
    from clip_retrieval.clip_front import add_static_endpoints

    add_static_endpoints(app, default_backend, None, url_column)

    api = Api(app)
    api.add_resource(MetricsSummary, "/metrics-summary")
    api.add_resource(
        IndicesList,
        "/indices-list",
        resource_class_kwargs={"indices": list(clip_resources.keys())},
    )
    api.add_resource(
        MetadataService,
        "/metadata",
        resource_class_kwargs={"clip_resources": clip_resources},
    )
    api.add_resource(
        KnnService, "/knn-service", resource_class_kwargs={"clip_resources": clip_resources}
    )
    CORS(app)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--indices_paths", type=str)
    parser.add_argument("--enable_hdf5", action="store_true")
    parser.add_argument("--enable_faiss_memory_mapping", action="store_true")
    parser.add_argument("--columns_to_return", nargs="*")
    parser.add_argument("--reorder_metadata_by_ivf_index", action="store_true")
    parser.add_argument("--default_backend", type=str, default=None)
    parser.add_argument("--url_column", type=str, default="url")
    parser.add_argument("--enable_mclip_option", action="store_true")
    parser.add_argument("--clip_model", type=str, default="RN50")
    parser.add_argument("--use_jit", action="store_true")
    parser.add_argument("--use_arrow", action="store_true")
    parser.add_argument("--provide_safety_model", action="store_true")
    parser.add_argument("--provide_violence_detector", action="store_true")
    parser.add_argument("--provide_aesthetic_embeddings", action="store_true")

    args = parser.parse_args()
    clip_back(**vars(args))
