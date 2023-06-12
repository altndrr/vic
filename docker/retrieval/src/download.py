import tarfile
from pathlib import Path

import gdown

RETRIEVAL_DATABASES = {
    "cc12m": "https://drive.google.com/uc?id=1HyM4mnKSxF0sqzAe-KZL8y-cQWRPiuXn&confirm=t",
    "english_words": "https://drive.google.com/uc?id=197poGaUJVP1Mh1qPL5yaNrYJuvd3JRb-&confirm=t",
    "pmd_top5": "https://drive.google.com/uc?id=15SDIf7KM8VIG_AxdnKkL1ODr_igOuZSD&confirm=t",
    "wordnet": "https://drive.google.com/uc?id=1q_StrVCnj8fPgvghXw-fSxp4qaSe0xvk&confirm=t",
}


def prepare_data(artifact_dir: str):
    """Download data if needed."""
    databases_path = Path(artifact_dir, "models", "databases")

    for name, url in RETRIEVAL_DATABASES.items():
        database_path = Path(databases_path, name)
        if database_path.exists():
            continue

        # download data
        target_path = Path(databases_path, name + ".tar.gz")
        try:
            gdown.download(url, str(target_path), quiet=False)
            tar = tarfile.open(target_path, "r:gz")
            tar.extractall(target_path.parent)
            tar.close()
            target_path.unlink()
        except FileNotFoundError:
            print(f"Could not download {url}.")
            print(f"Please download it manually and place it in {target_path.parent}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts/")

    args = parser.parse_args()
    prepare_data(args.artifact_dir)
