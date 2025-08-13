import os
import math
import requests
from pathlib import Path
from typing import List
from tqdm import tqdm

REDPAJAMA_FILES = [
    "config.json",
    "generation_config.json",
    "pytorch_model-00001-of-00003.bin",
    "pytorch_model-00002-of-00003.bin",
    "pytorch_model-00003-of-00003.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]

BASE_URL = "https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Instruct/resolve/main/"


def download_redpajama(target_dir: Path) -> List[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for fname in REDPAJAMA_FILES:
        url = BASE_URL + fname
        dest = target_dir / fname
        if dest.exists():
            downloaded.append(dest)
            continue
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        chunk = 1024 * 1024
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=fname) as pbar:
            for part in resp.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
                    pbar.update(len(part))
        downloaded.append(dest)
    return downloaded
