import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


DATASET_FILE = (
    Path(__file__).resolve().parents[2]
    / "mmdet3d_plugin/datasets/navsim_openscene_nuplan.py"
)
DETECTOR_FILE = (
    Path(__file__).resolve().parents[2]
    / "mmdet3d_plugin/navformer/detectors/navformer.py"
)


def _load_online_scoring_method():
    tree = ast.parse(DATASET_FILE.read_text())
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NavSimOpenSceneE2E"
    )
    method = next(
        node
        for node in dataset_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_compute_online_pdm_scores"
    )
    namespace = {
        "np": np,
        "logger": SimpleNamespace(info=lambda *_: None, warning=lambda *_: None),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), DATASET_FILE, "exec"), namespace)
    return namespace["_compute_online_pdm_scores"]


def _load_dataset_method(method_name):
    tree = ast.parse(DATASET_FILE.read_text())
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NavSimOpenSceneE2E"
    )
    method = next(
        node
        for node in dataset_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), DATASET_FILE, "exec"), namespace)
    return namespace[method_name]


def _method_source(file_path, class_name, method_name):
    source = file_path.read_text()
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method)


def _install_module(monkeypatch, name, **attributes):
    parts = name.split(".")
    for index in range(1, len(parts) + 1):
        module_name = ".".join(parts[:index])
        if module_name not in sys.modules:
            module = ModuleType(module_name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, module_name, module)
    module = sys.modules[name]
    for key, value in attributes.items():
        setattr(module, key, value)


def test_online_pdm_uses_official_evaluation_sampling(monkeypatch):
    calls = {}

    class TrajectorySampling:
        def __init__(self, num_poses, interval_length):
            self.num_poses = num_poses
            self.interval_length = interval_length

    class Trajectory:
        def __init__(self, poses):
            self.poses = poses

    class PDMSimulator:
        def __init__(self, proposal_sampling):
            calls["simulator_sampling"] = proposal_sampling

    class PDMScorer:
        def __init__(self, proposal_sampling):
            calls["scorer_sampling"] = proposal_sampling

    def pdm_score(**kwargs):
        calls["pdm_score"] = kwargs
        return SimpleNamespace(
            no_at_fault_collisions=0.91,
            drivable_area_compliance=0.82,
            ego_progress=0.73,
            time_to_collision_within_bound=0.64,
            comfort=0.55,
            score=0.46,
        )

    _install_module(
        monkeypatch,
        "nuplan.planning.simulation.trajectory.trajectory_sampling",
        TrajectorySampling=TrajectorySampling,
    )
    _install_module(
        monkeypatch,
        "navsim.common.dataclasses",
        Trajectory=Trajectory,
    )
    _install_module(
        monkeypatch,
        "navsim.evaluate.pdm_score",
        pdm_score=pdm_score,
    )
    _install_module(
        monkeypatch,
        "navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator",
        PDMSimulator=PDMSimulator,
    )
    _install_module(
        monkeypatch,
        "navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer",
        PDMScorer=PDMScorer,
    )

    result = {
        "token": "scene-token",
        "trajectory": np.arange(120, dtype=np.float32).reshape(40, 3),
        "score": float("nan"),
    }
    dataset = SimpleNamespace(
        metric_cache_dict={"scene-token": "cache-path"},
        get_metric_cache=lambda token: f"cache:{token}",
    )

    _load_online_scoring_method()(dataset, [result])

    model_trajectory = calls["pdm_score"]["model_trajectory"]
    future_sampling = calls["pdm_score"]["future_sampling"]
    assert model_trajectory.poses.shape == (8, 3)
    assert future_sampling.num_poses == 40
    assert future_sampling.interval_length == 0.1
    assert calls["simulator_sampling"] is future_sampling
    assert calls["scorer_sampling"] is future_sampling
    assert result["score"] == 0.46
    assert result["drivable_area_compliance"] == 0.82


def test_generated_trajectory_heads_defer_scores_to_online_evaluation():
    source = _method_source(DETECTOR_FILE, "NAVFormer", "forward_test")
    assert "requires_online_pdm_scoring" in source
    assert "use_online_pdm" in source
    assert "float('nan')" in source
    assert "DiffusionPlanningHead" not in source
    assert "GoalFlowPlanningHead" not in source


def test_generated_results_require_official_rescoring():
    requires_official = _load_dataset_method("_requires_official_pdm_rescoring")
    dataset = SimpleNamespace()

    assert requires_official(dataset, [{"score": float("nan")}]) is True
    assert requires_official(dataset, [{"score": 0.75}]) is False
    assert requires_official(
        dataset,
        [{"score": 0.75}, {"score": float("nan")}],
    ) is True


def test_evaluate_skips_embedded_scoring_for_generated_results():
    source = _method_source(DATASET_FILE, "NavSimOpenSceneE2E", "evaluate")

    assert "_requires_official_pdm_rescoring" in source
    assert "_compute_online_pdm_scores" in source
    assert source.index("_requires_official_pdm_rescoring") < source.index(
        "_compute_online_pdm_scores"
    )
    assert "DiffusionPlanningHead" not in source
    assert "GoalFlowPlanningHead" not in source


def test_evaluate_exports_submission_for_navtest_failures_subset():
    source = _method_source(DATASET_FILE, "NavSimOpenSceneE2E", "evaluate")

    assert '"navtest_failures_filtered.yaml"' in source
    assert "_navsim_submission.pkl" in source
