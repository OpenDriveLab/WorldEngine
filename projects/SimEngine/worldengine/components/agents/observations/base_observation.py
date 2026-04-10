from abc import ABC
from copy import deepcopy

from worldengine.engine.engine_utils import get_engine

import logging
logger = logging.getLogger(__name__)


class BaseObservation(ABC):
    """
    Observation: images / lidar / etc. information of different agents.
    """
    def __init__(self, agent):
        # assert not engine_initialized(), "Observations can not be created after initializing the simulation"
        self.current_observation = None
        self.agent = agent

    @property
    def engine(self):
        return get_engine()

    @property
    def observation_space(self):
        raise NotImplementedError

    def observe(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, env, vehicle=None):
        pass

    def destroy(self):
        """
        Clear allocated memory
        """
        pass
        # Config.clear_nested_dict(self.config)
        # self.config = None
