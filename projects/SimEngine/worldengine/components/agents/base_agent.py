"""
Base class for vehicle / pedestrian / etc. objects.
"""
import copy
import math
import numpy as np
import logging
from collections import deque

from abc import ABC
from worldengine.base_class.base_runnable import BaseRunnable
from worldengine.utils.type import WorldEngineObjectType
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD

from worldengine.components.agents.vehicle_model.build_vehicle import build_vehicle

# Navigation information, for IDM agents or SDC agent
from worldengine.components.agents.navigation.build_navigation import build_navigation

# Observation information, sensor, navigation, etc. Used for policy.
from worldengine.components.agents.observations.build_observations import build_observation

# policy functions.
from worldengine.components.agents.policy.build_policy import build_policy
from worldengine.components.agents.controller.build_controller import build_controller
from worldengine.components.agents.client.build_client import build_client

from worldengine.utils import math_utils
from worldengine.common.dataclasses import Trajectory

logger = logging.getLogger(__name__)


class BaseAgent(BaseRunnable, WorldEngineObjectType, ABC):
    """
    BaseAgent is something interacting with game engine.
    """

    # properties to normalize state information.
    MAX_LENGTH = 10
    MAX_WIDTH = 2.5
    MAX_HEIGHT = 2.5
    MAX_SPEED = 200
    MAX_STEERING = np.pi / 3  # in rad

    def __init__(self, object_id, object_track, name, random_seed=None, config=None, traj_step=0):
        config = copy.deepcopy(config)
        assert name == object_id
        BaseRunnable.__init__(self, name, random_seed, config)
        WorldEngineObjectType.__init__(self)

        # initial agent via <object_track>.
        self.object_track = object_track
        self._traj_step = traj_step
        
        self.vehicle = build_vehicle(object_id, self.config)
        if self.vehicle:
            self._length = self.vehicle.length
            self._height = self.vehicle.height
            self._width = self.vehicle.width
        else:
            self._length = self.object_track['length'][self._traj_step][0]
            self._height = self.object_track['height'][self._traj_step][0]
            self._width = self.object_track['width'][self._traj_step][0]

        # navigation. lane route information
        if self.config.get('agent_policy') == 'trajectory_policy' and object_id != 'ego':
            self.navigation = None
        elif self.config.get('ego_policy') == 'trajectory_policy':
            self.navigation = build_navigation(object_id, self.config)(self)
        else:
            self.navigation = build_navigation(object_id, self.config)(self)

        # observation space of the agent.
        #  it can be used for ego_agent.
        if self.config.get('agent_policy') == 'trajectory_policy' and object_id != 'ego':
            self.observation = None
        elif self.config.get('ego_policy') == 'trajectory_policy':
            self.observation = None
        else:
            self.observation = build_observation(object_id)(self)

        # policy of the agent.
        self.policy = build_policy(object_id, self.config)(self)

        # controller of the agent.
        self.controller = build_controller(object_id, self.config)(self)

        # client of the agent.
        self.client = build_client(object_id, self.config)(self)

        self._cur_pos = self.object_track[SD.POSITION][self._traj_step]
        self._cur_heading_theta = self.object_track[SD.HEADING][self._traj_step]
        self._cur_velocity = self.object_track['velocity'][self._traj_step]
        self._cur_angular_velocity = self.object_track['angular_velocity'][self._traj_step]

    def reset(self):
        if self.navigation:
            self.navigation.reset()

        # other info
        self.last_position = self._cur_pos
        self.last_heading_dir = self._cur_heading_theta
        self.last_velocity = self._cur_velocity  # 2D vector

    def before_step(self, action=None):
        """
        Save info and make decision before action
        """
        assert action is None

        self.last_position = self._cur_pos
        self.last_heading_dir = self._cur_heading_theta
        self.last_velocity = self._cur_velocity  # 2D vector
        return dict()

    def step(self):
        # obs = self.observation.observe() # Not to be used for now

        if self.policy.is_current_step_valid:
            lower_action = self.policy.act()
            if isinstance(lower_action, Trajectory):
                self.trajectory = lower_action
            elif isinstance(lower_action, list):
                self.lower_action = lower_action
            else:
                raise ValueError(f"Action type {type(lower_action)} is not supported")

            # controller to update the position.
            self.controller.step()

            self._traj_step += 1

    def after_step(self):
        if self.navigation:
            self.navigation.update_localization()

        step_info = {
            "position": self.current_position,
            "heading": self.current_heading,
            "velocity": float(self.current_speed),
            "length": self.length,
            "width": self.width,
            "height": self.height,
        } # TODO: valid judgement

        return step_info

    def set_policy(self, policy):
        pass

    def set_observation(self, observation):
        pass

    def convert_to_local_coordinates(self, pos: np.ndarray):
        """
        Give <pos> in world coordinates, and convert it to object coordinates.

        Args:
            pos: A numpy array with shape as [N, 2], indicating a set of point position
                to be normalized into the local coordinates.

        Return:
            local_pos
        """
        new_pos = math_utils.rotate_points(
            pos, self.current_position, -self.current_heading
        )
        return new_pos

    def convert_to_world_coordinates(self, pos: np.ndarray):
        """
        Given <pos> in local coordinates:
            x: heading distance. back (negative) / forward (positive)
            y: lateral distance. left (negative) / right (positive)

        Transform them into global coordinates.
        """
        new_pos = math_utils.rotate_points(pos, 0, self.current_heading)
        new_pos = new_pos + self.current_position
        return new_pos

    def destroy(self):
        pass

    def set_position(self, position):
        assert len(position) == 2 or len(position) == 3
        self._cur_pos = position

    @property
    def current_position(self):
        # return x/y coordinates in the global coordinates.
        return self._cur_pos[:2]

    def set_heading_theta(self, heading_theta, in_rad=True) -> None:
        """
        Set heading theta for this object
        :param heading_theta: float
        :param in_rad: when set to True, heading theta should be in rad, otherwise, in degree
        """
        if not in_rad:
            heading_theta = heading_theta / 180 * np.pi
        self._cur_heading_theta = heading_theta

    @property
    def current_heading(self):
        return self._cur_heading_theta

    def set_velocity(self, direction, value=None, in_local_frame=False):
        """
        Set velocity for object including the direction of velocity and the value (speed)
        The direction of velocity will be normalized automatically, value decided its scale
        :param direction: 2d array or list
        :param value: speed [m/s]
        :param in_local_frame: True, apply speed to local fram
        """
        if in_local_frame:
            direction = self.convert_to_world_coordinates(direction, [0, 0])

        if value is not None:
            norm_ratio = value / (math_utils.norm(direction[0], direction[1]) + 1e-6)
        else:
            norm_ratio = 1

        self._cur_velocity = direction * norm_ratio

    @property
    def current_velocity(self):
        return self._cur_velocity

    @property
    def velocity_km_h(self):
        velocity = self.current_velocity
        velocity_km_h = velocity * 3.6
        return velocity_km_h


    def set_angular_velocity(self, angular_velocity, in_rad=True):
        if not in_rad:
            angular_velocity = angular_velocity / 180 * np.pi
        self._cur_angular_velocity = angular_velocity

    @property
    def current_angular_velocity(self):
        return self._cur_angular_velocity

    @property
    def traj_step(self):
        return self._traj_step

    @property
    def current_speed(self):
        """
        return the speed in m/s
        """
        velocity = self.current_velocity
        speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        return min(max(speed, 0.0), 100000.0)

    @property
    def speed_km_h(self):
        """
        return the speed in km/h
        """
        velocity = self.current_velocity
        speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        speed = speed * 3.6
        return min(max(speed, 0.0), 100000.0)

    @property
    def height(self):
        return self._height

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    @property
    def bounding_box(self):
        """
        return the bounding box of the agent.
        """
        return np.concatenate(
            [self.current_position, [self.length, self.width, self.height, self.current_heading]]
        )

    @property
    def destination(self):
        # return x/y coordinates in the global coordinates.
        dest = self.config.get('destination', None)

        if dest is None:
            raise NotImplementedError('Destination must be set!!!')
            # use the last valid trajectory route as the dest.
            # pos = self.object_track[SD.POSITION]
            # valid = np.where(self.object_track[SD.VALID] == 1)
            # dest = pos[valid][-1]

        return dest[:2]

    def heading_diff(self, target_lane):
        """ cosine similarity between the lane orientation and heading. """

        lateral_dir = target_lane.lateral_direction(
            target_lane.local_coordinates(self.current_position)[0])
        lateral_dir_norm = math_utils.norm(lateral_dir[0], lateral_dir[1])

        agent_dir = [math.cos(self.current_heading),
                     math.sin(self.current_heading)]
        agent_dir_norm = math_utils.norm(agent_dir[0], agent_dir[1])

        if not lateral_dir_norm * agent_dir_norm:
            return 0
        cos = (
            (agent_dir[0] * lateral_dir[0] + agent_dir[1] * lateral_dir[1]) /
            (lateral_dir_norm * agent_dir_norm)
        )
        return math_utils.clip(cos, -1.0, 1.0) / 2 + 0.5
