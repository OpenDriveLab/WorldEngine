"""
Ego-state observation.
"""

import numpy as np

from worldengine.components.agents.observations.base_observation import BaseObservation
from worldengine.components.parameter_space import Box
from worldengine.utils import math_utils


class EgoStateObservation(BaseObservation):
    """
    Vehicle state info, navigation info,
    """
    # ego object size (3)
    # navigation information (3):
    #   lateral_to_left_lane / lateral_to_right_lane / lateral to current_lane.
    # ego state information (5)
    #   heading_diff_to_lane, speed, steering, last-, current-action.
    ego_state_obs_dim = 3 + 3 + 5

    # used to normalize the state_observation.
    LATERAL_DIST_NORM_SCALAR = 100

    def __init__(self, agent):
        super(EgoStateObservation, self).__init__(agent)

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

        # 1. ego shape observation.
        shape_info = [
            math_utils.clip(self.agent.length / self.agent.MAX_LENGTH, 0, 1),
            math_utils.clip(self.agent.height / self.agent.MAX_HEIGHT, 0, 1),
            math_utils.clip(self.agent.width / self.agent.MAX_WIDTH, 0, 1),
        ]
        info += shape_info

        # 2. lane line information.
        #  distance to left && right road boundary.
        lateral_to_left, lateral_to_right, = self.get_dist_to_left_right_lane()
        lateral_to_left /= self.LATERAL_DIST_NORM_SCALAR
        lateral_to_right /= self.LATERAL_DIST_NORM_SCALAR
        info += [math_utils.clip((lateral_to_left + 1) / 2, 0.0, 1.0),
                 math_utils.clip((lateral_to_right + 1) / 2, 0.0, 1.0)]
        #  distance to the exact lane.
        _, lateral = self.agent.navigation.current_lane.local_coordinates(self.agent.current_position)
        info.append(math_utils.clip(
            (lateral / self.LATERAL_DIST_NORM_SCALAR + 1) / 2, 0.0, 1.0))

        # 3. add navigation information,
        #  running on which lane etc.
        current_reference_lane = self.agent.navigation.current_ref_lanes[0]
        info += [

            # The angular difference between vehicle's heading and the lane heading at this location.
            self.agent.heading_diff(current_reference_lane),

            # The velocity of target vehicle
            math_utils.clip(
                (self.agent.current_speed / self.agent.MAX_SPEED + 1) / 2, 0.0, 1.0),

            # Current steering
            math_utils.clip(
                (self.agent.current_steering / self.agent.MAX_STEERING + 1) / 2, 0.0, 1.0),

            # The normalized actions at last steps
            math_utils.clip((self.agent.last_and_current_action[1][0] + 1) / 2, 0.0, 1.0),
            math_utils.clip((self.agent.last_and_current_action[1][1] + 1) / 2, 0.0, 1.0)
        ]

        navi_info = self.agent.navigation.navi_info
        ret_info = np.concatenate([np.array(info), navi_info])
        return ret_info

    def get_dist_to_left_right_lane(self):
        """
        Distance of the current agent to the left lane and right lane.
        """
        navigation = self.agent.navigation

        if navigation is None or navigation.current_ref_lanes is None:
            return 0, 0

        current_reference_lane = navigation.current_ref_lanes[0]
        _, lateral_to_reference = current_reference_lane.local_coordinates(
            self.agent.current_position)
        lateral_to_left = (
            navigation.get_left_lateral_range() +
            navigation.get_current_lane_width() / 2 -
            lateral_to_reference)
        lateral_to_right = navigation.get_current_lateral_range() - lateral_to_left
        return lateral_to_left, lateral_to_right
