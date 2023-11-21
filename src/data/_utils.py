import tarfile
import zipfile
from pathlib import Path

import gdown
import requests
from rich.progress import track

from src import utils

log = utils.get_logger(__name__)


def download_data(url: str, target: Path, from_gdrive: bool = False) -> None:
    """Download data from a URL.

    Args:
        url (str): The URL to download the data from.
        target (Path): The path to save the data to.
        from_gdrive (bool): Whether the data is from Google Drive.
    """
    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=False)

    if from_gdrive:
        gdown.download(url, str(target), quiet=False)
    else:
        with requests.get(url, stream=True, timeout=10.0) as r:
            r.raise_for_status()
            chunk_size = 8192
            p_bar = track(
                r.iter_content(chunk_size=chunk_size),
                total=int(r.headers["Content-Length"]) / chunk_size,
                description=f"Downloading data to {target}",
            )
            with open(target, "wb") as f:
                for chunk in p_bar:
                    f.write(chunk)


def extract_data(target: Path) -> None:
    """Extract data from an archive.

    Supported formats: zip, tar, tar.gz.

    Args:
        target (Path): The path to the file to extract.
    """
    if target.name.endswith(".zip"):
        zip_ref = zipfile.ZipFile(target, "r")
        zip_ref.extractall(target.parent)
        zip_ref.close()
    elif target.name.endswith(".tar"):
        tar = tarfile.open(target, "r:")
        tar.extractall(target.parent)
        tar.close()
    elif target.name.endswith(".tar.gz") or target.name.endswith(".tgz"):
        tar = tarfile.open(target, "r:gz")
        tar.extractall(target.parent)
        tar.close()
    else:
        raise NotImplementedError(f"Unsupported file format: {target.suffix}")
