"""
Local configuration utilities to replace DigitalTwin dependency.
"""
import os
import pickle
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Tuple


@dataclass
class RoadBlockConfig:
    """Road block configuration base class"""

    road_block_name: str
    road_block: Tuple = (331250, 4690950, 331350, 4691050)
    data_root: str = ""
    city: Literal['sg-one-north', 'us-ma-boston', 'us-na-las-vegas-strip', 'us-pa-pittsburgh-hazelwood'] = 'us-ma-boston'
    interval: int = 1
    expand_buffer: int = 30
    reconstruct_buffer: int = 10
    selected_videos: Tuple = field(default_factory=tuple)
    split: Literal['trainval', 'test', 'all'] = 'trainval'
    collect_raw: bool = False
    exclude_bad_registration: bool = True
    use_colmap_ba: bool = False


@dataclass
class CentralConfig(RoadBlockConfig):
    """Central configuration class"""

    central_log: str = ""
    central_tokens: List[str] = field(default_factory=list)
    multi_traversal_mode: Literal['reconstruction', 'registration', 'off'] = 'off'


def _build_config_loader():
    """Build a YAML loader that maps legacy nuplan_scripts classes to local ones."""
    loader = yaml.Loader

    _TAG_MAP = {
        'tag:yaml.org,2002:python/object:nuplan_scripts.utils.config.CentralConfig': CentralConfig,
        'tag:yaml.org,2002:python/object:nuplan_scripts.utils.config.RoadBlockConfig': RoadBlockConfig,
    }

    for tag, cls in _TAG_MAP.items():
        def _constructor(loader, node, _cls=cls):
            fields = loader.construct_mapping(node, deep=True)
            return _cls(**fields)
        loader.add_constructor(tag, _constructor)

    return loader


_config_loader = _build_config_loader()


def load_config(config_path: str):
    """
    Load configuration file.

    Args:
        config_path: Path to configuration file, supports .yaml or .yml files

    Returns:
        RoadBlockConfig or CentralConfig object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix in ['.yml', '.yaml']:
        with open(config_path, 'r') as f:
            config_dict = yaml.load(f, Loader=_config_loader)
        return config_dict
    else:
        raise NotImplementedError(f"Unsupported config file format: {config_path.suffix}")


class VideoScene:
    """
    Video scene wrapper class, simplified version.
    """

    def __init__(self, config):
        """
        Initialize VideoScene.

        Args:
            config: Configuration object (RoadBlockConfig or CentralConfig)
        """
        self.config = config
        self.video_scene_dict = None

    def load_pickle(self, path: str, verbose: bool = True):
        """
        Load pickle file.

        Args:
            path: Path to pickle file
            verbose: Whether to print logs

        Returns:
            Loaded video_scene_dict
        """
        if verbose:
            print(f'Loading pickle from {path}')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Pickle file not found: {path}")

        with open(path, 'rb') as f:
            self.video_scene_dict = pickle.load(f)

        return self.video_scene_dict

    @property
    def name(self):
        """Return scene name"""
        return self.config.road_block_name if hasattr(self.config, 'road_block_name') else 'unknown'
