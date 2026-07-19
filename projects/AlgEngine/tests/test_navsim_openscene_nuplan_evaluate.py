import numpy as np
import pytest

from mmdet3d_plugin.datasets.navsim_openscene_nuplan import NavSimOpenSceneE2E


def test_selection_mode_requires_vocabulary_pdm_cache(tmp_path):
    dataset = NavSimOpenSceneE2E.__new__(NavSimOpenSceneE2E)
    dataset.diffusiondrive_data_mode = False
    dataset.pdm_path = str(tmp_path / "missing_pdm_cache")

    with pytest.raises(FileNotFoundError, match="PDM score cache not found"):
        dataset.load_pdm_infos()


def test_generated_trajectory_mode_skips_vocabulary_pdm_warnings(caplog, tmp_path):
    dataset = NavSimOpenSceneE2E.__new__(NavSimOpenSceneE2E)
    dataset.diffusiondrive_data_mode = True
    dataset.pdm_path = str(tmp_path / "missing_pdm_cache")

    dataset.load_pdm_infos()
    sample = dataset.get_pdm_score_info({"sample_idx": "test-token"})

    assert dataset.pdm_dict == {}
    assert sample["score"].shape == (8192,)
    assert not any("PDM score" in record.message for record in caplog.records)


def test_evaluate_accepts_none_logger_for_official_rescoring(tmp_path):
    dataset = NavSimOpenSceneE2E.__new__(NavSimOpenSceneE2E)
    dataset.nav_filter_path = "navtest_failures.yaml"
    results = [
        {
            "token": "test-token",
            "trajectory": np.zeros((44, 3), dtype=np.float32),
            "ade_4s": 0.0,
            "fde_4s": 0.0,
            "no_at_fault_collisions": np.nan,
            "drivable_area_compliance": np.nan,
            "ego_progress": np.nan,
            "time_to_collision_within_bound": np.nan,
            "comfort": np.nan,
            "score": np.nan,
        }
    ]
    output_prefix = tmp_path / "evaluation"

    metrics = dataset.evaluate(
        results,
        logger=None,
        jsonfile_prefix=str(output_prefix),
    )

    assert metrics["ade_4s"] == 0.0
    assert output_prefix.with_suffix(".csv").is_file()
