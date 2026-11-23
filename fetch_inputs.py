"""Fetch the pinned WiCE dataset files without adding them to the source tree."""
import argparse
import hashlib
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

REVISION = "ddeb6c183665e2a20c5f03c5aa07f03888b9870f"
BASE = f"https://raw.githubusercontent.com/ryokamoi/wice/{REVISION}/data/entailment_retrieval/subclaim/"
DIGESTS = {
    "train": "e8ba1ed589e22c5f9d41c8d858dd10bdc17665d224f8ce28be9d9024239cf513",
    "dev": "4747a5a60c85c0c12cbf1cc7a2ab4bc9279fd4fd01a83d36630b7151330b6f57",
    "test": "3202b7bff5979dfbca745f48edd8da5ad166da0886fd15042cc1ab9d2010ee19",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(path, split):
    if path.exists():
        if digest(path) != DIGESTS[split]:
            raise ValueError(f"{path} is not the pinned version")
        return
    with NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
    try:
        with urlopen(BASE + path.name, timeout=60) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        if digest(temporary) != DIGESTS[split]:
            raise ValueError(f"Checksum mismatch downloading {path.name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in DIGESTS:
        if split == "test" and args.development_only:
            continue
        fetch(args.output / f"{split}.jsonl", split)
