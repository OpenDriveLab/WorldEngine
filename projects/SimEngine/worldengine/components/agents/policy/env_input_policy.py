from worldengine.engine.engine_utils import get_global_config
import numpy as np

from worldengine.components.agents.policy.base_policy import BasePolicy
from worldengine.utils.math_utils import clip
from worldengine.common.dataclasses import Trajectory


class EnvInputPolicy(BasePolicy):
    DEBUG_MARK_COLOR = (252, 119, 3, 255)

    def __init__(self, agent, config = None, random_seed = None):
        # Since control object may change
        super(EnvInputPolicy, self).__init__(agent=agent, config=config, random_seed=random_seed)
        

    def act(self):
        action = self.engine.external_actions
        if isinstance(action, Trajectory):
            return action
        elif isinstance(action, list):
            return action
        else:
            raise ValueError(f"Action type {type(action)} is not supported")
    
    @property
    def is_current_step_valid(self):
        return True

