import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "merge_navsim_pdm_score_shards.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("merge_navsim_pdm_score_shards", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_score(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["", "token", "valid", "score"])
        writer.writeheader()
        writer.writerows(rows)


def test_merge_score_files_recomputes_token_weighted_average(tmp_path):
    module = _load_module()
    shard_0 = tmp_path / "shard_0.csv"
    shard_1 = tmp_path / "shard_1.csv"
    _write_score(
        shard_0,
        [
            {"": "0", "token": "a", "valid": "True", "score": "0.25"},
            {"": "1", "token": "b", "valid": "True", "score": "0.75"},
            {"": "2", "token": "average", "valid": "True", "score": "0.5"},
        ],
    )
    _write_score(
        shard_1,
        [
            {"": "0", "token": "c", "valid": "True", "score": "1.0"},
            {"": "1", "token": "average", "valid": "True", "score": "1.0"},
        ],
    )

    output = module.merge_score_files([shard_0, shard_1], tmp_path / "merged.csv")
    rows = list(csv.DictReader(output.open()))

    assert [row["token"] for row in rows] == ["a", "b", "c", "average"]
    assert float(rows[-1]["score"]) == pytest.approx(2 / 3)
    assert rows[-1]["valid"] == "True"


def test_merge_score_files_rejects_duplicate_tokens(tmp_path):
    module = _load_module()
    shard_0 = tmp_path / "shard_0.csv"
    shard_1 = tmp_path / "shard_1.csv"
    row = {"": "0", "token": "same", "valid": "True", "score": "0.5"}
    _write_score(shard_0, [row])
    _write_score(shard_1, [row])

    with pytest.raises(ValueError, match="duplicate token"):
        module.merge_score_files([shard_0, shard_1], tmp_path / "merged.csv")
