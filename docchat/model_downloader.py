"""Utilities for ensuring a full local copy of a Hugging Face model.

Previous revision attempted ad‑hoc HTTP GET of a hard‑coded RedPajama file list. Large
weight shards are stored with Git LFS and direct resolve URLs returned 404, leading to
partial, brittle local states and repeated remote downloads by Transformers.

We now rely on huggingface_hub.snapshot_download to materialise a complete local snapshot
of the repo the first time, then always re‑use those files offline (no symlinks) on later runs.
"""

from pathlib import Path
from typing import Optional, Tuple
import logging
import json

from huggingface_hub import snapshot_download, hf_hub_download  # noqa: F401 (hf_hub_download reserved for future granular needs)

LOGGER = logging.getLogger("DocChat.Downloader")

def _verify_snapshot(dir_path: Path) -> Tuple[bool, str]:
    """Deep integrity check for an HF snapshot directory.

    Verifies presence of index (if sharded), referenced shard files and approximate size.
    Returns (is_ok, message).
    """
    try:
        if not dir_path.exists():
            return False, "directory missing"
        # Basic required files
        config_present = any(dir_path.glob("*config.json"))
        tokenizer_present = (dir_path / "tokenizer.json").exists() or (dir_path / "tokenizer.model").exists()
        weight_files = list(dir_path.glob("*.safetensors")) + list(dir_path.glob("pytorch_model-*.bin"))
        single_weight_file = (dir_path / "pytorch_model.bin")
        if single_weight_file.exists():
            weight_files.append(single_weight_file)
        if not config_present:
            return False, "missing config.json"
        if not weight_files:
            return False, "no weight files found"
        # If sharded index exists, validate coverage and size.
        index_file = dir_path / "pytorch_model.bin.index.json"
        if index_file.exists():
            data = json.loads(index_file.read_text(encoding="utf-8"))
            weight_map = data.get("weight_map", {})
            metadata = data.get("metadata", {})
            total_size_expected = metadata.get("total_size")
            referenced_shards = set(weight_map.values())
            # Ensure all referenced shard files exist
            missing = [s for s in referenced_shards if not (dir_path / s).exists()]
            if missing:
                return False, f"missing shard files: {missing[:3]}{'...' if len(missing)>3 else ''}"
            # Compute actual size
            actual = sum((dir_path / s).stat().st_size for s in referenced_shards)
            if isinstance(total_size_expected, (int, float)):
                # Allow 1% or 5MB tolerance
                tolerance = max(5 * 1024 * 1024, int(total_size_expected * 0.01))
                if abs(actual - int(total_size_expected)) > tolerance:
                    return False, f"size mismatch actual={actual} expected={int(total_size_expected)}"
        # Lightweight sanity: tokenizer presence helps speed but not strictly required
        if not tokenizer_present:
            return False, "missing tokenizer.json/model"
        return True, "ok"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"verification error: {e}"


def ensure_local_model(repo_id: str, local_dir: Path, allow_patterns: Optional[list[str]] = None, exclude_patterns: Optional[list[str]] = None, revision: Optional[str] = None, force: bool = False, verify: bool = True) -> Path:
    """Ensure a local snapshot of the HF repo exists inside local_dir.

    Args:
        repo_id: Hugging Face repository id (e.g. togethercomputer/RedPajama-INCITE-7B-Instruct)
        local_dir: Target directory for the snapshot.
        allow_patterns: Optional whitelist of glob patterns (kept broad / None downloads everything).
        exclude_patterns: Optional blacklist patterns.
        revision: Optional git revision / branch / tag.
        force: If True, re-download (clears existing dir first).

    Returns:
        Path to the local directory containing the snapshot (usable as model path).
    """
    local_dir = local_dir.expanduser().resolve()
    if force and local_dir.exists():
        # Cautious removal: only delete if directory contains expected model artifacts
        try:
            for p in local_dir.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted([d for d in local_dir.rglob("*") if d.is_dir()], reverse=True):
                p.rmdir()
            local_dir.rmdir()
        except Exception as e:
            LOGGER.warning("Force re-download cleanup failed: %s", e)
    if local_dir.exists() and not force:
        if verify:
            ok, msg = _verify_snapshot(local_dir)
            if ok:
                LOGGER.info("Using existing verified local model snapshot at %s", local_dir)
                return local_dir
            else:
                LOGGER.warning("Local model snapshot at %s failed integrity check (%s); re-downloading.", local_dir, msg)
                force = True  # trigger cleanup path below
        else:
            LOGGER.info("Using existing local model snapshot at %s (verification skipped)", local_dir)
            return local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading model snapshot repo=%s -> %s (this may take a while the first time)", repo_id, local_dir)
    try:
        # Note: local_dir_use_symlinks deprecated (HF Hub >=0.23); pass only supported args
        snapshot_download(
            repo_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=exclude_patterns,
            revision=revision,
        )
    except Exception as e:
        LOGGER.exception("Snapshot download failed for %s", repo_id)
        raise RuntimeError(f"Failed to snapshot download {repo_id}: {e}")
    # Post-download verification (fail fast if corruption)
    if verify:
        ok, msg = _verify_snapshot(local_dir)
        if not ok:
            LOGGER.error("Downloaded snapshot failed integrity verification (%s)", msg)
            raise RuntimeError(f"Corrupted or incomplete snapshot for {repo_id}: {msg}")
    LOGGER.info("Model snapshot complete and verified.")
    return local_dir

# Backwards compatibility shim for older code paths
def download_redpajama(target_dir: Path):  # pragma: no cover - kept for legacy imports
    return ensure_local_model("togethercomputer/RedPajama-INCITE-7B-Instruct", target_dir)
