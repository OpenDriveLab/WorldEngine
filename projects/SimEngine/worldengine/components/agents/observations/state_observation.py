"""
State-observation for all components participating in the traffic flow.
"""

import numpy as np

from worldengine.components.agents.observations.base_observation import BaseObservation
from worldengine.components.parameter_space import Box
from worldengine.utils import math_utils


class StateObservation(BaseObservation):
    """
    Vehicle state info, navigation info,
    """
    # object size (3)
    # navigation information (2):
    #   current_position (x, y)
    # object state information (3)
    #   heading, vx, vy
    ego_state_obs_dim = 3 + 2 + 3

    # used to normalize global coordinates.
    GLOBAL_COORD_NORM_SCALAR = 10000

    def __init__(self, agent):
        super(StateObservation, self).__init__(agent)

        self.navi_info_dim = self.agent.navigation.get_navigation_info_dim()

    @property
    def observation_space(self):
        shape = self.ego_state_obs_dim + self.navi_info_dim
        return Box(-0.0, 1.0, shape=(shape, ), dtype=np.float32)

    def observe(self):
        """
        Return the observation of target vehicle.
        Ego states: [
                    shape information,

                    [Distance to left yellow Continuous line,
                    Distance to right Side Walk]

                    Difference of heading between ego vehicle and current lane,
                    Current speed,
                    Current steering,
                    Throttle/brake of last frame,
                    Steering of last frame,
                    Yaw Rate,

                     [Lateral Position on current lane.], if use lane_line detector, else:
                     [lane_line_detector cloud points]
                    ], dim >= 9
        :param agent: BaseAgent
        :return: Vehicle State + Navigation information
        """

        info = []

        # 1. vehicle shape observation.
        shape_info = [
            math_utils.clip(self.agent.length / self.agent.MAX_LENGTH, 0, 1),
            math_utils.clip(self.agent.height / self.agent.MAX_HEIGHT, 0, 1),
            math_utils.clip(self.agent.width / self.agent.MAX_WIDTH, 0, 1),
        ]
        info += shape_info

        # 2. vehicle position information.
        position_info = [
            math_utils.clip(
                (self.agent.current_position[0] / self.GLOBAL_COORD_NORM_SCALAR + 1) / 2, 0, 1),
            math_utils.clip(
                (self.agent.current_position[1] / self.GLOBAL_COORD_NORM_SCALAR + 1) / 2, 0, 1),
        ]
        info += position_info

        # 3. vehicle speed and heading information.
        info += [
            math_utils.clip(
                (math_utils.wrap_to_pi(self.agent.current_heading) / np.pi + 1) / 2, 0, 1
            ),
            math_utils.clip(
                (self.agent.current_velocity[0] / self.agent.MAX_SPEED + 1) / 2, 0, 1
            ),
            math_utils.clip(
                (self.agent.current_velocity[1] / self.agent.MAX_SPEED + 1) / 2, 0, 1
            ),
        ]

        navi_info = self.agent.navigation.navi_info
        ret_info = np.concatenate([np.array(info), navi_info])
        return ret_info
