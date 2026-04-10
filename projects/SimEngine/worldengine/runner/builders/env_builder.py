import logging
from typing import List, Optional, Dict, Union
from tqdm import tqdm
from omegaconf import DictConfig

from worldengine.envs.build_env import build_env
from worldengine.utils.multithreading.worker_pool import WorkerPool

logger = logging.getLogger(__name__)

def distribute_scenes(cfg: DictConfig, scene_dict: Dict[str, dict], worker: WorkerPool) -> Dict[str, List[dict]]:
    """
    Distribute scenes based on areas.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param scene_dict: Dict[str, dict]. Dictionary of scenes.
    :return: Dict[str, List[dict]]. Dictionary of scenes distributed.
    """

    scene_dict_distributed = dict()
    if cfg.distributed_mode == 'SINGLE_NODE':
        return {'main_thread': scene_dict}
    elif cfg.distributed_mode == 'SCENARIO_BASED':
        num_workers = worker.config.number_of_gpus_per_node
        items = list(scene_dict.items())
        total = len(items)
        base_size = total // num_workers
        remainder = total % num_workers
        start_idx = 0
        for split_idx in range(num_workers):
            current_size = base_size + (1 if split_idx < remainder else 0)
            if current_size == 0:
                continue
            else:
                end_idx = start_idx + current_size
                scene_dict_distributed[f'{cfg.worker_id_prefix}{split_idx}'] = dict(items[start_idx:end_idx])
                start_idx = end_idx
    else:
        raise ValueError(f'Not supported distributed_mode {cfg.distributed_mode}')

    return scene_dict_distributed

def build_envs(
    cfg: DictConfig,
    worker: Union[WorkerPool, None],
    scene_dict: Dict[str, dict],
    callbacks_worker: Optional[WorkerPool] = None
):
    """
    Build simulations.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param callbacks: Callbacks for simulation.
    :param worker: Worker for job execution.
    :param callbacks_worker: worker pool to use for callbacks from sim
    :param pre_built_planners: List of pre-built planners to run in simulation.
    :return A dict of simulation engines with challenge names.
    """
    logger.info('Building environments...')

    # Create Simulation object container
    envs = list()

    # TODO: Filtter scenes
    # scene_filter = None
    # scenes = scene_filter(scenes=scenes)

    logger.info('Building simulations from %d scenes...', len(scene_dict))
    scene_dict_distributed = distribute_scenes(cfg, scene_dict, worker=worker)
    for name, info_dicts in tqdm(scene_dict_distributed.items(), desc='Loading envs'):
        env = build_env(cfg, name=name, data=info_dicts)
        envs.append(env)

    logger.info('Building environments...DONE!')
    return envs
