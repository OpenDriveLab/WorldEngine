import importlib.util
import pickle
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "prepare_navsim_metric_cache_index.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_navsim_metric_cache_index", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_submission(path, tokens):
    with path.open("wb") as file:
        pickle.dump({"predictions": [{token: None for token in tokens}]}, file)


def _write_cache_metadata(root, token_paths):
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    metadata_file = metadata / "source.csv"
    metadata_file.write_text("file_name\n" + "\n".join(str(path) for path in token_paths) + "\n")


def test_prepare_index_writes_exact_submission_paths(tmp_path):
    module = _load_module()
    cache_root = tmp_path / "cache"
    path_a = cache_root / "log" / "unknown" / "token_a" / "metric_cache.pkl"
    path_b = cache_root / "log" / "unknown" / "token_b" / "metric_cache.pkl"
    path_c = cache_root / "log" / "unknown" / "token_c" / "metric_cache.pkl"
    _write_cache_metadata(cache_root, [path_a, path_b, path_c])
    submission = tmp_path / "submission.pkl"
    _write_submission(submission, ["token_b", "token_a"])

    output = module.prepare_index(submission, cache_root, tmp_path / "filtered")

    assert output.read_text().splitlines() == ["file_name", str(path_b), str(path_a)]


def test_prepare_index_rejects_missing_cache(tmp_path):
    module = _load_module()
    cache_root = tmp_path / "cache"
    path_a = cache_root / "log" / "unknown" / "token_a" / "metric_cache.pkl"
    _write_cache_metadata(cache_root, [path_a])
    submission = tmp_path / "submission.pkl"
    _write_submission(submission, ["token_a", "missing_token"])

    with pytest.raises(module.MissingMetricCacheError, match="1 submission tokens missing metric cache"):
        module.prepare_index(submission, cache_root, tmp_path / "filtered")


def test_prepare_index_does_not_stat_every_cache_payload(tmp_path, monkeypatch):
    module = _load_module()
    cache_root = tmp_path / "cache"
    cache_path = cache_root / "log" / "unknown" / "token_a" / "metric_cache.pkl"
    _write_cache_metadata(cache_root, [cache_path])
    submission = tmp_path / "submission.pkl"
    _write_submission(submission, ["token_a"])

    def reject_resolve(*args, **kwargs):
        raise AssertionError("cache payload paths must not be resolved or stat'ed")

    monkeypatch.setattr(module.Path, "resolve", reject_resolve)
    output = module.prepare_index(submission, cache_root, tmp_path / "filtered")

    assert output.read_text().splitlines() == ["file_name", str(cache_path)]


def test_prepare_index_writes_balanced_isolated_shards(tmp_path):
    module = _load_module()
    cache_root = tmp_path / "cache"
    paths = [
        cache_root / "log" / "unknown" / f"token_{index}" / "metric_cache.pkl"
        for index in range(5)
    ]
    _write_cache_metadata(cache_root, paths)
    submission = tmp_path / "submission.pkl"
    _write_submission(submission, [f"token_{index}" for index in range(5)])

    output_root = tmp_path / "filtered"
    module.prepare_index(submission, cache_root, output_root, num_shards=2)

    shard_0 = output_root / "shards" / "0" / "cache" / "metadata" / "metric_cache.csv"
    shard_1 = output_root / "shards" / "1" / "cache" / "metadata" / "metric_cache.csv"
    assert shard_0.read_text().splitlines() == [
        "file_name",
        str(paths[0]),
        str(paths[2]),
        str(paths[4]),
    ]
    assert shard_1.read_text().splitlines() == [
        "file_name",
        str(paths[1]),
        str(paths[3]),
    ]


def test_prepare_index_removes_stale_shards(tmp_path):
    module = _load_module()
    cache_root = tmp_path / "cache"
    paths = [
        cache_root / "log" / "unknown" / f"token_{index}" / "metric_cache.pkl"
        for index in range(4)
    ]
    _write_cache_metadata(cache_root, paths)
    submission = tmp_path / "submission.pkl"
    _write_submission(submission, [f"token_{index}" for index in range(4)])

    output_root = tmp_path / "filtered"
    module.prepare_index(submission, cache_root, output_root, num_shards=4)
    stale_shard = output_root / "shards" / "3"
    assert stale_shard.is_dir()

    module.prepare_index(submission, cache_root, output_root, num_shards=2)

    assert sorted(path.name for path in (output_root / "shards").iterdir()) == [
        "0",
        "1",
    ]
