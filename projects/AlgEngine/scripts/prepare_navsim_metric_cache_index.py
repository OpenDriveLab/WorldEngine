#!/usr/bin/env python3
"""Prepare a submission-filtered metadata index for NAVSIM's official scorer."""

import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


class MissingMetricCacheError(RuntimeError):
    """Raised when a submission token has no corresponding metric cache."""


def _load_submission_tokens(submission_path: Path) -> List[str]:
    with submission_path.open("rb") as file:
        submission = pickle.load(file)

    predictions = submission.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise ValueError("submission must contain exactly one predictions mapping")
    if not isinstance(predictions[0], dict):
        raise ValueError("submission predictions entry must be a token mapping")
    return list(predictions[0])


def _load_metric_cache_paths(metric_cache_path: Path) -> Dict[str, Path]:
    metadata_dir = metric_cache_path / "metadata"
    metadata_files = sorted(metadata_dir.glob("*.csv"))
    if len(metadata_files) != 1:
        raise ValueError(
            f"expected exactly one metadata CSV under {metadata_dir}, "
            f"found {len(metadata_files)}"
        )

    lines = metadata_files[0].read_text().splitlines()[1:]
    cache_paths: Dict[str, Path] = {}
    for line in lines:
        if not line.strip():
            continue
        cache_path = Path(line.strip())
        if not cache_path.is_absolute():
            cache_path = metric_cache_path / cache_path
        # Metadata may contain tens of thousands of cache payloads on GPFS.
        # Path.resolve() performs an lstat for each component of each path,
        # although the scorer only needs the path strings. Normalize without
        # touching every payload on disk.
        cache_paths[cache_path.parent.name] = Path(os.path.abspath(cache_path))
    return cache_paths


def _write_metadata(output_dir: Path, cache_paths: Iterable[Path]) -> Path:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_path = metadata_dir / "metric_cache.csv"
    rows = ["file_name", *(str(path) for path in cache_paths)]
    output_path.write_text("\n".join(rows) + "\n")
    return output_path


def _write_shard_indexes(
    output_dir: Path,
    cache_paths: Sequence[Path],
    num_shards: int,
) -> List[Path]:
    """Write balanced cache indexes for isolated scorer processes."""

    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not cache_paths:
        raise ValueError("cannot shard an empty metric-cache index")

    shard_count = min(num_shards, len(cache_paths))
    shard_paths = []
    for shard_index in range(shard_count):
        shard_cache_paths = cache_paths[shard_index::shard_count]
        shard_paths.append(
            _write_metadata(
                output_dir / "shards" / str(shard_index) / "cache",
                shard_cache_paths,
            )
        )
    return shard_paths


def prepare_index(
    submission_path: Path,
    metric_cache_path: Path,
    output_dir: Path,
    num_shards: int = 1,
) -> Path:
    """Write an official-compatible cache index containing submission tokens."""

    submission_path = Path(submission_path)
    metric_cache_path = Path(metric_cache_path)
    output_dir = Path(output_dir)

    tokens = _load_submission_tokens(submission_path)
    cache_paths = _load_metric_cache_paths(metric_cache_path)
    missing = [token for token in tokens if token not in cache_paths]
    if missing:
        examples = ", ".join(missing[:5])
        raise MissingMetricCacheError(
            f"{len(missing)} submission tokens missing metric cache; examples: {examples}"
        )

    selected_paths = [cache_paths[token] for token in tokens]
    output_path = _write_metadata(output_dir, selected_paths)
    shards_dir = output_dir / "shards"
    if shards_dir.exists():
        shutil.rmtree(shards_dir)
    if num_shards > 1:
        shard_paths = _write_shard_indexes(output_dir, selected_paths, num_shards)
        print(f"Prepared {len(shard_paths)} isolated metric-cache shards")
    print(f"Prepared {len(tokens)} metric-cache entries at {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare metric-cache metadata filtered to a NAVSIM submission."
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare_index(
        args.submission,
        args.metric_cache,
        args.output_dir,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
