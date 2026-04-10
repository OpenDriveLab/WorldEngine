import time
import logging
from collections import OrderedDict
from typing import Dict, AnyStr
from omegaconf import DictConfig
import numpy as np

from worldengine.common.randomizable import Randomizable
from worldengine.utils.utils import concat_step_infos

logger = logging.getLogger(__name__)

class BaseEngine(Randomizable):
    global_random_seed = None

    def __init__(self, global_config: DictConfig):
        self.global_config = global_config
        Randomizable.__init__(self, self.global_random_seed)

        self.episode_step = 0

        self._managers = OrderedDict()

        self.external_actions = None
        self.seed(0)

    def get_sensor(self):
        if 'render_manager' in self._managers:
            return self._managers['render_manager'].get_observations()
        else:
            return {}

    def reset(self) -> Dict:
        """
        Clear and generate the whole scene
        """
        step_infos = {}

        # initialize
        self._episode_start_time = time.time()
        self.episode_step = 0

        # reset manager
        for manager_name, manager in self._managers.items():
            # clean all manager
            new_step_infos = manager.before_reset()
            step_infos = concat_step_infos([step_infos, new_step_infos])
        self._object_clean_check()

        for manager_name, manager in self._managers.items():
            new_step_infos = manager.reset()
            step_infos = concat_step_infos([step_infos, new_step_infos])

        for manager_name, manager in self._managers.items():
            new_step_infos = manager.after_reset()
            step_infos = concat_step_infos([step_infos, new_step_infos])

        return step_infos

    def before_step(self, external_actions: Dict[AnyStr, np.array]):
        """
        Entities make decision here, and prepare for step
        All entities can access this global manager to query or interact with others
        :param external_actions: Dict[agent_id:action]
        :return:
        """
        step_infos = {}
        self.external_actions = external_actions
        for manager in self.managers.values():
            new_step_infos = manager.before_step()
            step_infos = concat_step_infos([step_infos, new_step_infos])
        return step_infos

    def step(self, step_num: int = 1) -> None:
        """
        Step the dynamics of each entity on the road.
        :param step_num: Decision of all entities will repeat *step_num* times
        """
        self.episode_step += 1 # In order to align with rendering.
        # episode_step = 0 means initialization and other nums means iteration.
        
        for i in range(step_num):
            # simulate or replay
            for name, manager in self.managers.items():
                manager.step()

    def after_step(self, *args, **kwargs) -> Dict:
        """
        Update states after finishing movement
        :return: if this episode is done
        """
        step_infos = {}
        for manager in self.managers.values():
            new_step_info = manager.after_step(*args, **kwargs)
            step_infos = concat_step_infos([step_infos, new_step_info])
        return step_infos

    def dump_episode(self, pkl_file_name=None) -> None:
        """Dump the data of an episode."""
        pass

    def close(self):
        """
        Note:
        Instead of calling this func directly, close Engine by using engine_utils.close_engine
        """

        # destroy managers.
        if len(self._managers) > 0:
            for name, manager in self._managers.items():
                setattr(self, name, None)
                if manager is not None:
                    manager.destroy()

    def __del__(self):
        logger.debug("{} is destroyed".format(self.__class__.__name__))

    # initialization function.
    def register_manager(self, manager_name: str, manager):
        """
        Add a manager to BaseEngine, then all objects can communicate with this class
        :param manager_name: name shouldn't exist in self._managers and not be same as any class attribute
        :param manager: subclass of BaseManager
        """
        assert manager_name not in self._managers, "Manager already exists in BaseEngine, Use update_manager() to " \
                                                   "overwrite"
        assert not hasattr(self, manager_name), "Manager name can not be same as the attribute in BaseEngine"
        self._managers[manager_name] = manager
        setattr(self, manager_name, manager)
        self._managers = OrderedDict(sorted(self._managers.items(), key=lambda k_v: k_v[-1].PRIORITY))

    def seed(self, random_seed):
        self.global_random_seed = random_seed
        super(BaseEngine, self).seed(random_seed)
        for mgr in self._managers.values():
            mgr.seed(random_seed)

    @property
    def current_track_agent(self):
        agent = self._managers['agent_manager'].get_dynamic_agents
        return agent

    @property
    def agents(self):
        agents = self._managers['agent_manager'].all_agents
        return agents

    def setup_main_camera(self):
        pass

    @property
    def current_seed(self):
        return self.global_random_seed

    @property
    def global_seed(self):
        return self.global_random_seed

    def _object_clean_check(self):
        pass

    def update_manager(self, manager_name: str, manager, destroy_previous_manager=True):
        """
        Update an existing manager with a new one
        :param manager_name: existing manager name
        :param manager: new manager
        """
        assert manager_name in self._managers, "You may want to call register manager, since {} is not in engine".format(
            manager_name
        )
        existing_manager = self._managers.pop(manager_name)
        if destroy_previous_manager:
            existing_manager.destroy()
        self._managers[manager_name] = manager
        setattr(self, manager_name, manager)
        self._managers = OrderedDict(sorted(self._managers.items(), key=lambda k_v: k_v[-1].PRIORITY))

    # properties from managers.
    @property
    def managers(self):
        # whether to froze other managers
        return self._managers

    @property
    def current_scene(self):
        return self._managers['scenario_manager'].current_scene

    @property
    def current_map(self):
        return self._managers['map_manager'].current_map

    @property
    def sim_time_interval(self):
        """ Each step interval of the simulator,
        For nuPlan, this should be 0.05 * sample_rate (sample_rate is 10 at 2Hz)
        """
        return self.current_scene['sample_rate'] * 0.05

    @property
    def num_scenarios(self):
        return self._managers['scenario_manager'].num_scenarios
