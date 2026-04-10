import logging
import os
from typing import Optional
from omegaconf import DictConfig

from worldengine.engine.base_engine import BaseEngine

# Module-level variables for process isolation
# Each process has its own copy of these variables
_engine_instance: Optional[BaseEngine] = None
_global_config: Optional[DictConfig] = None


def initialize_engine(env_global_config: DictConfig):
    """
    Initialize the engine core. Each process should only have at most one instance of the engine.

    Args:
        env_global_config: the global config.

    Returns:
        The engine.
    """
    global _engine_instance

    if _engine_instance is None:
        _engine_instance = BaseEngine(env_global_config)
    else:
        raise PermissionError(
            f"There should be only one BaseEngine instance in one process. "
            f"PID: {os.getpid()}"
        )
    return _engine_instance


def get_engine() -> Optional[BaseEngine]:
    """Get the engine instance for the current process."""
    return _engine_instance


def get_object(object_name):
    return get_engine().get_objects([object_name])


def engine_initialized() -> bool:
    """Check if the engine is initialized in the current process."""
    return _engine_instance is not None


def close_engine():
    """Close and cleanup the engine instance for the current process."""
    global _engine_instance

    if _engine_instance is not None:
        _engine_instance.close()
        _engine_instance = None


def get_global_config() -> Optional[DictConfig]:
    """Get the global config for the current process."""
    return _global_config


def initialize_global_config(global_config: DictConfig):
    """
    Initialize the global config for the current process.
    You can preset the engine config before launching the engine.
    """
    global _global_config

    if engine_initialized():
        raise RuntimeError("Can not call initialize_global_config after engine initialization!")
    _global_config = global_config
