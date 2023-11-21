from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset

__all__ = ["ImageDataset"]


class ImageDataset(VisionDataset):
    """Dataset for images.

    If only the root directory is provided, the dataset works as the `ImageFolder` dataset from
    torchvision. It is otherwise possible to provide a list of images and/or labels. To modify
    the class names, it is possible to provide a list of class names. If the class names are not
    provided, they are inferred from the folder names.

    Args:
        root (str): Root directory of dataset where `images` are found.
        images (list[str], optional): List of images. Defaults to None.
        labels (list[int] | list[list[int]], optional): List of labels (supports multi-labels).
            Defaults to None.
        class_names (list[str], optional): List of class names. Defaults to None.
        classes_to_idx (list[str], optional): Mapping from class names to class indices. Defaults
            to None.
        transform (Callable | list[Callable], optional): A function/transform that takes in a
            PIL image and returns a transformed version. If a list of transforms is provided, they
            are applied depending on the target label. Defaults to None.
        target_transform (Callable, optional): A function/transform that takes in the target and
            transforms it. Defaults to None.
        return_id (bool, optional): If `True`, the dataset will return also the image path of the
            sample. Defaults to `True`.

     Attributes:
        class_names (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index, domain_index) tuples.
        images (list): List of paths to images.
        targets (list): The class_index value for each image in the dataset.
    """

    def __init__(
        self,
        root: str,
        images: Optional[list[str]] = None,
        labels: Optional[Union[list[int], list[list[int]]]] = None,
        class_names: Optional[list[str]] = None,
        classes_to_idx: Optional[dict[str, int]] = None,
        transform: Optional[Union[Callable, list[Callable]]] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        if not images:
            images = [str(path) for path in Path(root).glob("*/*")]

        if not class_names:
            class_names = {Path(f).parent.name for f in images}

        if not labels:
            folder_names = {Path(f).parent.name for f in images}
            folder_names = sorted(folder_names)
            folder_names_to_idx = {c: i for i, c in enumerate(folder_names)}
            labels = [folder_names_to_idx[Path(f).parent.name] for f in images]

        self.samples = list(zip(images, labels))
        self.images = images
        self.targets = labels

        self.class_names = class_names
        self.classes_to_idx = classes_to_idx or {c: i for i, c in enumerate(self.class_names)}

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.loader = default_loader

    def __getitem__(self, index: int) -> dict:
        path, targets_idx = self.samples[index]
        targets_idx = [targets_idx] if isinstance(targets_idx, int) else targets_idx
        targets_name = [self.class_names[t] for t in targets_idx]

        targets = torch.tensor(targets_idx, dtype=torch.long)
        image_pil = self.loader(path)
        if self.transform is not None:
            if isinstance(self.transform, list):
                image_tensor = self.transform[targets](image_pil)
            else:
                image_tensor = self.transform(image_pil)
        if self.target_transform is not None:
            targets = [self.target_transform(t) for t in targets]

        targets_one_hot = torch.zeros(len(self.class_names), dtype=torch.long)
        targets_one_hot[targets] = 1

        data = {
            "images_fp": path,
            "images_pil": image_pil,
            "images_tensor": image_tensor,
            "targets_idx": targets_idx,
            "targets_one_hot": targets_one_hot,
            "targets_name": targets_name,
        }
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        if self.class_names is not None:
            if len(self.class_names) > 10:
                body += [f"Classes: {', '.join(self.class_names[:10])}..."]
            else:
                body += [f"Classes: {', '.join(self.class_names)}"]
        if hasattr(self, "transform") and self.transform is not None:
            body += [repr(self.transform)]
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)


def default_loader(path: str) -> Any:
    """Loads an image from a path.

    Args:
        path (str): str to the image.

    Returns:
        PIL.Image: The image.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
