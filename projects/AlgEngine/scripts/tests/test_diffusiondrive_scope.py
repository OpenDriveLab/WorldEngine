import ast
import os
import runpy
from pathlib import Path


ALGENGINE_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = (
    ALGENGINE_ROOT / "mmdet3d_plugin/datasets/navsim_openscene_nuplan.py"
)
CONFIG_FILES = [
    ALGENGINE_ROOT / "configs/multi_4node/e2e_diffusiondrive.py",
    ALGENGINE_ROOT / "configs/navformer/e2e_diffusiondrive.py",
    ALGENGINE_ROOT / "configs/diffusiondrive/e2e_diffusiondrive.py",
]
VARIANT_CONFIGS = {
    "e2e_diffusiondrive_100pct.py": ("navtrain.yaml", 50),
    "e2e_diffusiondrive_13pct.py": ("navtrain_13pct.yaml", 16),
    "e2e_diffusiondrive_25pct.py": ("navtrain_25pct.yaml", 16),
    "e2e_diffusiondrive_50pct.py": ("navtrain_50pct.yaml", 16),
    "e2e_diffusiondrive_60pct.py": ("navtrain_60pct.yaml", 16),
    "e2e_diffusiondrive_70pct.py": ("navtrain_70pct.yaml", 16),
    "e2e_diffusiondrive_80pct.py": ("navtrain_80pct.yaml", 16),
    "e2e_diffusiondrive_90pct.py": ("navtrain_90pct.yaml", 16),
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


def test_diffusiondrive_configs_share_the_canonical_configuration():
    canonical = CONFIG_FILES[0].read_text()

    assert all(path.read_text() == canonical for path in CONFIG_FILES[1:])
    assert canonical.count("diffusiondrive_data_mode=True") == 3


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
