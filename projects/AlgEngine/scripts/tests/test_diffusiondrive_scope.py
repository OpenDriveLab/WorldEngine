import ast
import os
import runpy
from pathlib import Path


ALGENGINE_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = (
    ALGENGINE_ROOT / "mmdet3d_plugin/datasets/navsim_openscene_nuplan.py"
)
TRACKED_DIFFUSIONDRIVE_CONFIGS = [
    ALGENGINE_ROOT / "configs/navformer/e2e_diffusiondrive.py",
    ALGENGINE_ROOT / "configs/diffusiondrive/e2e_diffusiondrive.py",
]
NAVFORMER_CONFIG_DIR = ALGENGINE_ROOT / "configs/navformer"
VARIANT_CONFIGS = {
    "e2e_diffusiondrive_100pct.py": ("navtrain.yaml", 100),
    "e2e_diffusiondrive_13pct.py": ("navtrain_13pct.yaml", 100),
    "e2e_diffusiondrive_25pct.py": ("navtrain_25pct.yaml", 100),
    "e2e_diffusiondrive_50pct.py": ("navtrain_50pct.yaml", 100),
    "e2e_diffusiondrive_60pct.py": ("navtrain_60pct.yaml", 100),
    "e2e_diffusiondrive_70pct.py": ("navtrain_70pct.yaml", 100),
    "e2e_diffusiondrive_80pct.py": ("navtrain_80pct.yaml", 100),
    "e2e_diffusiondrive_90pct.py": ("navtrain_90pct.yaml", 100),
}


def _dataset_method_source(method_name):
    source = DATASET_FILE.read_text()
    tree = ast.parse(source)
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
    return ast.get_source_segment(source, method)


def test_diffusiondrive_data_mode_is_opt_in():
    init_source = _dataset_method_source("__init__")
    update_source = _dataset_method_source("update_ego_prediction")

    assert "diffusiondrive_data_mode=False" in init_source
    assert "self.diffusiondrive_data_mode = diffusiondrive_data_mode" in init_source
    assert "if self.diffusiondrive_data_mode:" in update_source
    assert "sdc_status=navsim_status" in update_source
    assert "sdc_status=sdc_status[[0, 1, 6]]" in update_source


def test_tracked_diffusiondrive_configs_enable_navsim_data_mode():
    for path in TRACKED_DIFFUSIONDRIVE_CONFIGS:
        config = runpy.run_path(str(path))

        assert config["model"]["planning_head"]["type"] == "DiffusionPlanningHead"
        assert all(
            config["data"][split]["diffusiondrive_data_mode"] is True
            for split in ("train", "val", "test")
        )


def test_navformer_planners_use_stable_vadv2_perception_freezing():
    stable_model = runpy.run_path(str(NAVFORMER_CONFIG_DIR / "e2e_vadv2.py"))[
        "model"
    ]
    freeze_keys = (
        "freeze_img_backbone",
        "freeze_img_neck",
        "freeze_bn",
        "freeze_bev_encoder",
    )
    expected = {key: stable_model[key] for key in freeze_keys}

    model = runpy.run_path(
        str(NAVFORMER_CONFIG_DIR / "e2e_diffusiondrive.py")
    )["model"]

    assert {key: model[key] for key in freeze_keys} == expected


def test_diffusiondrive_variant_configs_preserve_dataset_splits():
    config_dir = ALGENGINE_ROOT / "configs/diffusiondrive"

    for filename, (expected_yaml, expected_epochs) in VARIANT_CONFIGS.items():
        config = runpy.run_path(str(config_dir / filename))
        model = config["model"]
        planning_head = model["planning_head"]
        optimizer = config["optimizer"]
        data = config["data"]

        assert os.path.basename(config["nav_filter_path_train"]) == expected_yaml
        assert config["total_epochs"] == expected_epochs
        assert config["evaluation"]["interval"] == 10
        assert model["freeze_img_neck"] is True
        assert model["freeze_bev_encoder"] is True
        assert planning_head["trajectory_loss_weight"] == 12.0
        assert optimizer["lr"] == 6e-4
        assert optimizer["weight_decay"] == 1e-4
        assert config["checkpoint_config"] == {
            "interval": 10,
            "max_keep_ckpts": 5,
        }
        assert all(
            data[split]["diffusiondrive_data_mode"] is True
            for split in ("train", "val", "test")
        )
